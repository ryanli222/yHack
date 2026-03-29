/*
 * mega_motor_control.ino
 * Autonomous Wheelchair — Mecanum Drive Controller
 * Arduino Mega 2560 + 4x IBT-2 Motor Drivers + Quadrature Encoders
 *
 * IBT-2 control: RPWM = forward PWM, LPWM = reverse PWM.
 *
 * Serial protocol (115200 baud, newline-delimited JSON):
 *   Receive: {"vx": float, "vy": float, "w": float}
 *   Send:    {"estop": 0|1, "enc": [dFL, dFR, dRL, dRR]}
 *            enc = delta ticks since last report (signed int32)
 *
 * Motor driver wiring (IBT-2 RPWM/LPWM):
 *   FL: RPWM=D13, LPWM=D12
 *   FR: RPWM=D4,  LPWM=D5
 *   RL: RPWM=D10, LPWM=D11
 *   RR: RPWM=D2,  LPWM=D3
 * Encoder wiring (quadrature, 1x decode on channel A):
 *   FL: A=D2(INT0), B=D3 — CONFLICT with RR motor pins; move if using encoders
 *   FR: A=D18(INT5), B=D19
 *   RL: A=D20(INT3), B=D21
 *   RR: A=D22(polled), B=D23
 * E-stop: D32 (active LOW, polled)
 *
 * Dependencies: ArduinoJson (v7.x)
 */

#include <ArduinoJson.h>

// ============================================================
// 1. PIN DEFINITIONS
// ============================================================

// Motor FL (Front-Left)  — driver 1
#define PIN_FL_RPWM  13
#define PIN_FL_LPWM  12

// Motor FR (Front-Right) — driver 3
#define PIN_FR_RPWM   4
#define PIN_FR_LPWM   5

// Motor RL (Rear-Left)   — driver 2
#define PIN_RL_RPWM  10
#define PIN_RL_LPWM  11

// Motor RR (Rear-Right)  — driver 4
// NOTE: D2/D3 were previously FL encoder A/B. If encoders are wired,
//       move FL encoder to free pins (e.g. D34/D35) and update below.
#define PIN_RR_RPWM   2
#define PIN_RR_LPWM   3

// E-stop button (active LOW, polled — D32 has no external INT on Mega)
#define PIN_ESTOP     32

// Encoder channel A pins (interrupt-driven for FL/FR/RL, polled for RR)
#define PIN_ENC_FL_A   2   // INT0
#define PIN_ENC_FL_B   3
#define PIN_ENC_FR_A  18   // INT5
#define PIN_ENC_FR_B  19
#define PIN_ENC_RL_A  20   // INT3
#define PIN_ENC_RL_B  21
#define PIN_ENC_RR_A  22   // polled
#define PIN_ENC_RR_B  23

// ============================================================
// 2. CONFIGURATION
// ============================================================

// Motor direction signs: flip to +1 or -1 after hardware test.
//                          FL   FR   RL   RR
const int MOTOR_SIGN[4] = { +1,  +1,  +1,  +1 };

// Encoder direction signs: flip if ticks go negative during forward motion.
//                            FL   FR   RL   RR
const int ENC_SIGN[4]   = { +1,  +1,  +1,  +1 };

#define PWM_MAX              15   // TESTING: just enough to verify motors respond. Raise to 60+ once verified. Max is 255.
#define SERIAL_BAUD         115200
#define WATCHDOG_TIMEOUT_MS 500
#define LOOP_PERIOD_MS       10   // ~100 Hz
#define SERIAL_BUF_SIZE     128

// ============================================================
// 3. INTERNAL STATE
// ============================================================

enum Motor { FL = 0, FR = 1, RL = 2, RR = 3 };

const uint8_t pinRPWM[4] = { PIN_FL_RPWM, PIN_FR_RPWM, PIN_RL_RPWM, PIN_RR_RPWM };
const uint8_t pinLPWM[4] = { PIN_FL_LPWM, PIN_FR_LPWM, PIN_RL_LPWM, PIN_RR_LPWM };

int motorCommand[4] = { 0, 0, 0, 0 };

bool estopActive = false;

unsigned long lastCommandTime = 0;

char serialBuf[SERIAL_BUF_SIZE];
uint8_t serialBufIdx = 0;

// ---- Encoder counters (volatile, written from ISRs) ----
volatile int32_t encCount[4] = { 0, 0, 0, 0 };

// Last sent counts — we report delta ticks each loop
int32_t encLast[4] = { 0, 0, 0, 0 };

// Last known state of RR channel A (for polling)
uint8_t rrLastA = 0;

// ============================================================
// 4. ENCODER ISRs
// ============================================================

void encISR_FL() {
  if (digitalRead(PIN_ENC_FL_B) == HIGH) encCount[FL] += ENC_SIGN[FL];
  else                                    encCount[FL] -= ENC_SIGN[FL];
}

void encISR_FR() {
  if (digitalRead(PIN_ENC_FR_B) == HIGH) encCount[FR] += ENC_SIGN[FR];
  else                                    encCount[FR] -= ENC_SIGN[FR];
}

void encISR_RL() {
  if (digitalRead(PIN_ENC_RL_B) == HIGH) encCount[RL] += ENC_SIGN[RL];
  else                                    encCount[RL] -= ENC_SIGN[RL];
}

// ============================================================
// 5. MOTOR CONTROL
// ============================================================

void setMotor(uint8_t index, int speed) {
  speed = speed * MOTOR_SIGN[index];
  if (speed > PWM_MAX)   speed = PWM_MAX;
  if (speed < -PWM_MAX)  speed = -PWM_MAX;

  if (speed > 0) {
    analogWrite(pinRPWM[index], (uint8_t)speed);
    analogWrite(pinLPWM[index], 0);
  } else if (speed < 0) {
    analogWrite(pinRPWM[index], 0);
    analogWrite(pinLPWM[index], (uint8_t)(-speed));
  } else {
    analogWrite(pinRPWM[index], 0);
    analogWrite(pinLPWM[index], 0);
  }
}

void stopAllMotors() {
  for (uint8_t i = 0; i < 4; i++) {
    analogWrite(pinRPWM[i], 0);
    analogWrite(pinLPWM[i], 0);
    motorCommand[i] = 0;
  }
}

// ============================================================
// 6. MECANUM INVERSE KINEMATICS
// ============================================================

void computeMecanumIK(float vx, float vy, float omega) {
  // X mecanum configuration — rollers diagonal is opposite to H/O drive.
  // FL/RR share +Vy; FR/RL share -Vy.
  float raw[4];
  raw[FL] = vx + vy - omega;
  raw[FR] = vx - vy + omega;
  raw[RL] = vx - vy - omega;
  raw[RR] = vx + vy + omega;

  float maxVal = 0.0f;
  for (uint8_t i = 0; i < 4; i++) {
    float absVal = fabs(raw[i]);
    if (absVal > maxVal) maxVal = absVal;
  }

  if (maxVal > 1e-6f) {
    float scale = (maxVal > 1.0f) ? (float)PWM_MAX / maxVal : (float)PWM_MAX;
    for (uint8_t i = 0; i < 4; i++) motorCommand[i] = (int)(raw[i] * scale);
  } else {
    for (uint8_t i = 0; i < 4; i++) motorCommand[i] = 0;
  }
}

// ============================================================
// 7. SERIAL PARSING
// ============================================================

bool processSerialLine(const char* line) {
  JsonDocument doc;
  if (deserializeJson(doc, line)) return false;
  if (!doc.containsKey("vx") || !doc.containsKey("vy") || !doc.containsKey("w")) return false;

  computeMecanumIK(doc["vx"].as<float>(), doc["vy"].as<float>(), doc["w"].as<float>());
  lastCommandTime = millis();
  return true;
}

void readSerial() {
  while (Serial.available() > 0) {
    char c = Serial.read();
    if (c == '\n' || c == '\r') {
      if (serialBufIdx > 0) {
        serialBuf[serialBufIdx] = '\0';
        processSerialLine(serialBuf);
        serialBufIdx = 0;
      }
    } else if (serialBufIdx < SERIAL_BUF_SIZE - 1) {
      serialBuf[serialBufIdx++] = c;
    } else {
      serialBufIdx = 0;
    }
  }
}

// ============================================================
// 8. SERIAL RESPONSE
// ============================================================

void sendStatus() {
  // Atomically snapshot delta ticks (disable interrupts briefly)
  int32_t snap[4];
  noInterrupts();
  for (uint8_t i = 0; i < 4; i++) snap[i] = encCount[i];
  interrupts();

  int32_t delta[4];
  for (uint8_t i = 0; i < 4; i++) {
    delta[i] = snap[i] - encLast[i];
    encLast[i] = snap[i];
  }

  JsonDocument doc;
  doc["estop"] = estopActive ? 1 : 0;
  JsonArray enc = doc["enc"].to<JsonArray>();
  for (uint8_t i = 0; i < 4; i++) enc.add(delta[i]);

  serializeJson(doc, Serial);
  Serial.println();
}

// ============================================================
// 9. SETUP
// ============================================================

void setup() {
  Serial.begin(SERIAL_BAUD);

  // Motor PWM output pins
  for (uint8_t i = 0; i < 4; i++) {
    pinMode(pinRPWM[i], OUTPUT);
    pinMode(pinLPWM[i], OUTPUT);
    analogWrite(pinRPWM[i], 0);
    analogWrite(pinLPWM[i], 0);
  }

  // E-stop (active LOW, internal pull-up, polled)
  pinMode(PIN_ESTOP, INPUT_PULLUP);
  estopActive = (digitalRead(PIN_ESTOP) == LOW);

  // Encoder channel A — interrupt pins
  pinMode(PIN_ENC_FL_A, INPUT_PULLUP);
  pinMode(PIN_ENC_FL_B, INPUT_PULLUP);
  pinMode(PIN_ENC_FR_A, INPUT_PULLUP);
  pinMode(PIN_ENC_FR_B, INPUT_PULLUP);
  pinMode(PIN_ENC_RL_A, INPUT_PULLUP);
  pinMode(PIN_ENC_RL_B, INPUT_PULLUP);

  attachInterrupt(digitalPinToInterrupt(PIN_ENC_FL_A), encISR_FL, RISING);
  attachInterrupt(digitalPinToInterrupt(PIN_ENC_FR_A), encISR_FR, RISING);
  attachInterrupt(digitalPinToInterrupt(PIN_ENC_RL_A), encISR_RL, RISING);

  // RR encoder — polled
  pinMode(PIN_ENC_RR_A, INPUT_PULLUP);
  pinMode(PIN_ENC_RR_B, INPUT_PULLUP);
  rrLastA = digitalRead(PIN_ENC_RR_A);

  lastCommandTime = millis();
  Serial.println("{\"status\":\"ready\"}");
}

// ============================================================
// 10. MAIN LOOP (~100 Hz)
// ============================================================

void loop() {
  unsigned long loopStart = millis();

  readSerial();

  // Poll e-stop
  estopActive = (digitalRead(PIN_ESTOP) == LOW);

  // Watchdog
  if ((millis() - lastCommandTime) > WATCHDOG_TIMEOUT_MS) {
    stopAllMotors();
  }

  // Poll RR encoder (channel A rising edge)
  uint8_t rrNowA = digitalRead(PIN_ENC_RR_A);
  if (rrNowA == HIGH && rrLastA == LOW) {
    // Rising edge detected
    if (digitalRead(PIN_ENC_RR_B) == HIGH) encCount[RR] += ENC_SIGN[RR];
    else                                    encCount[RR] -= ENC_SIGN[RR];
  }
  rrLastA = rrNowA;

  // Drive or stop motors
  if (estopActive) {
    stopAllMotors();
  } else {
    for (uint8_t i = 0; i < 4; i++) setMotor(i, motorCommand[i]);
  }

  sendStatus();

  // Maintain loop rate
  unsigned long elapsed = millis() - loopStart;
  if (elapsed < LOOP_PERIOD_MS) delay(LOOP_PERIOD_MS - elapsed);
}
