/*
 * mega_motor_control.ino
 * Autonomous Wheelchair — Mecanum Drive Controller
 * Arduino Mega 2560 + 4x IBT-2 Motor Drivers + Quadrature Encoders
 *
 * IBT-2 control: RPWM = forward PWM, LPWM = reverse PWM.
 *
 * Serial protocol (115200 baud, newline-delimited JSON):
 *   Receive: {"vx": float, "vy": float, "w": float}
 *            "q"            — e-stop (stop all motors)
 *            "r"            — resume (clear e-stop)
 *            "w"            — forward
 *            "s"            — backward
 *            "a"            — strafe left
 *            "d"            — strafe right
 *            "1"-"4"        — spin wheel CW  (FL, FR, RL, RR)
 *            "5"-"8"        — spin wheel CCW (FL, FR, RL, RR)
 *   Note: single-char movement commands run until watchdog cuts them (~500ms)
 *   Send:    {"estop": 0|1, "enc": [dFL, dFR, dRL, dRR]}
 *            enc = delta ticks since last report (signed int32)
 *
 * Motor driver wiring (IBT-2 RPWM/LPWM):
 *   FL: RPWM=D13, LPWM=D12
 *   FR: RPWM=D4,  LPWM=D5
 *   RL: RPWM=D10, LPWM=D11
 *   RR: RPWM=D2,  LPWM=D3
 * Encoder wiring (quadrature, interrupt-driven on CHN A rising edge):
 *   FL: CHN A=D18(INT5), CHN B=D22
 *   FR: CHN A=D19(INT4), CHN B=D23
 *   RL: CHN A=D20(INT3), CHN B=D52
 *   RR: CHN A=D21(INT2), CHN B=D53
 * E-stop: software only — send 'q' over serial
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
#define PIN_RR_RPWM   2
#define PIN_RR_LPWM   3

// Encoder CHN A pins (interrupt-driven — INT5/4/3/2)
#define PIN_ENC_FL_A  18   // INT5
#define PIN_ENC_FR_A  19   // INT4
#define PIN_ENC_RL_A  20   // INT3
#define PIN_ENC_RR_A  21   // INT2
// Encoder CHN B pins (direction sense, read inside ISR)
#define PIN_ENC_FL_B  22
#define PIN_ENC_FR_B  23
#define PIN_ENC_RL_B  52
#define PIN_ENC_RR_B  53

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

// Encoder counters (volatile, written from ISRs)
volatile int32_t encCount[4] = { 0, 0, 0, 0 };

// Last sent counts — we report delta ticks each loop
int32_t encLast[4] = { 0, 0, 0, 0 };

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

void encISR_RR() {
  if (digitalRead(PIN_ENC_RR_B) == HIGH) encCount[RR] += ENC_SIGN[RR];
  else                                    encCount[RR] -= ENC_SIGN[RR];
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

// Set one motor, zero the rest — used for per-wheel spin tests
void spinOneMotor(uint8_t idx, int speed) {
  for (uint8_t i = 0; i < 4; i++) motorCommand[i] = 0;
  motorCommand[idx] = speed;
}

bool processSerialLine(const char* line) {
  if (line[1] != '\0') goto json;  // more than one char → must be JSON

  switch (line[0]) {
    // ---- control ----
    case 'q':
      estopActive = true;
      stopAllMotors();
      return true;
    case 'r':
      estopActive = false;
      lastCommandTime = millis();
      return true;

    // ---- directional (wasd) ----
    case 'w': computeMecanumIK( 1.0f,  0.0f, 0.0f); lastCommandTime = millis(); return true;
    case 's': computeMecanumIK(-1.0f,  0.0f, 0.0f); lastCommandTime = millis(); return true;
    case 'a': computeMecanumIK( 0.0f, -1.0f, 0.0f); lastCommandTime = millis(); return true;
    case 'd': computeMecanumIK( 0.0f,  1.0f, 0.0f); lastCommandTime = millis(); return true;

    // ---- individual wheel CW (1-4 = FL, FR, RL, RR) ----
    case '1': spinOneMotor(FL,  PWM_MAX); lastCommandTime = millis(); return true;
    case '2': spinOneMotor(FR,  PWM_MAX); lastCommandTime = millis(); return true;
    case '3': spinOneMotor(RL,  PWM_MAX); lastCommandTime = millis(); return true;
    case '4': spinOneMotor(RR,  PWM_MAX); lastCommandTime = millis(); return true;

    // ---- individual wheel CCW (5-8 = FL, FR, RL, RR) ----
    case '5': spinOneMotor(FL, -PWM_MAX); lastCommandTime = millis(); return true;
    case '6': spinOneMotor(FR, -PWM_MAX); lastCommandTime = millis(); return true;
    case '7': spinOneMotor(RL, -PWM_MAX); lastCommandTime = millis(); return true;
    case '8': spinOneMotor(RR, -PWM_MAX); lastCommandTime = millis(); return true;
  }

json:
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

  // Encoder pins
  pinMode(PIN_ENC_FL_A, INPUT_PULLUP); pinMode(PIN_ENC_FL_B, INPUT_PULLUP);
  pinMode(PIN_ENC_FR_A, INPUT_PULLUP); pinMode(PIN_ENC_FR_B, INPUT_PULLUP);
  pinMode(PIN_ENC_RL_A, INPUT_PULLUP); pinMode(PIN_ENC_RL_B, INPUT_PULLUP);
  pinMode(PIN_ENC_RR_A, INPUT_PULLUP); pinMode(PIN_ENC_RR_B, INPUT_PULLUP);

  attachInterrupt(digitalPinToInterrupt(PIN_ENC_FL_A), encISR_FL, RISING);
  attachInterrupt(digitalPinToInterrupt(PIN_ENC_FR_A), encISR_FR, RISING);
  attachInterrupt(digitalPinToInterrupt(PIN_ENC_RL_A), encISR_RL, RISING);
  attachInterrupt(digitalPinToInterrupt(PIN_ENC_RR_A), encISR_RR, RISING);

  lastCommandTime = millis();
  Serial.println("{\"status\":\"ready\"}");
}

// ============================================================
// 10. MAIN LOOP (~100 Hz)
// ============================================================

void loop() {
  unsigned long loopStart = millis();

  readSerial();

  // Watchdog
  if ((millis() - lastCommandTime) > WATCHDOG_TIMEOUT_MS) {
    stopAllMotors();
  }

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
