"""
Voice interface: speech-to-text + text-to-speech for the wheelchair.

Uses ElevenLabs Python SDK for both STT and TTS.
Fallbacks: Whisper (STT), pyttsx3 (TTS).

Audio capture via sounddevice. Silence detection ends recording.

Usage:
  voice = create_voice_interface()
  await voice.start()
  transcript = await voice.listen()          # blocks until user speaks
  await voice.speak("On my way to the door") # plays through speakers
"""

from __future__ import annotations

import asyncio
import io
import logging
import os
import struct
import tempfile
import wave
from pathlib import Path
from typing import Optional

from config import load_config

logger = logging.getLogger(__name__)

# Load .env
try:
    from dotenv import load_dotenv
    _env = Path(__file__).resolve().parent.parent / ".env"
    if _env.exists():
        load_dotenv(_env)
except ImportError:
    pass

# Pre-cached common responses for low-latency TTS
RESPONSES = {
    "on_my_way": "On my way.",
    "arrived": "I've arrived.",
    "stopping": "Stopping now.",
    "dont_know": "I don't know where that is.",
    "rerouting": "Obstacle detected. Rerouting.",
    "listening": "I'm listening.",
}

SAMPLE_RATE = 16000
CHANNELS = 1
SILENCE_THRESHOLD = 500     # RMS threshold for silence
SILENCE_DURATION_S = 1.5    # seconds of silence to stop recording
MAX_RECORD_S = 10.0         # max recording duration


class VoiceInterface:
    """Real voice interface with ElevenLabs SDK + fallbacks."""

    def __init__(self):
        cfg = load_config().voice
        self._api_key = cfg.api_key or os.environ.get("ELEVENLABS_API_KEY", "")
        self._voice_id = cfg.voice_id
        self._stt_fallback = cfg.stt_fallback
        self._tts_fallback = cfg.tts_fallback

        self._stt_method: Optional[str] = None
        self._tts_method: Optional[str] = None
        self._eleven_client = None
        self._pyttsx_engine = None

    async def start(self) -> None:
        """Probe available STT/TTS backends."""
        # Try ElevenLabs SDK
        if self._api_key:
            try:
                from elevenlabs import ElevenLabs
                self._eleven_client = ElevenLabs(api_key=self._api_key)
                self._stt_method = "elevenlabs"
                self._tts_method = "elevenlabs"
                logger.info("STT: ElevenLabs SDK")
                logger.info("TTS: ElevenLabs SDK (voice=%s)", self._voice_id)
                return
            except ImportError:
                logger.warning("elevenlabs package not installed, trying fallbacks")

        # STT fallback
        try:
            import whisper  # noqa: F401
            self._stt_method = "whisper"
            logger.info("STT: Whisper (local)")
        except ImportError:
            self._stt_method = None
            logger.warning("STT: No backend available")

        # TTS fallback
        try:
            import pyttsx3
            self._pyttsx_engine = pyttsx3.init()
            self._tts_method = "pyttsx3"
            logger.info("TTS: pyttsx3 (local)")
        except Exception:
            self._tts_method = None
            logger.warning("TTS: No backend available")

    # ------------------------------------------------------------------
    # STT: listen for speech
    # ------------------------------------------------------------------

    async def listen(self) -> str:
        """Record audio from mic, transcribe, return text."""
        audio = await self._record_audio()
        if audio is None:
            return ""

        if self._stt_method == "elevenlabs":
            return await self._stt_elevenlabs(audio)
        elif self._stt_method == "whisper":
            return await self._stt_whisper(audio)
        else:
            logger.warning("No STT backend -- returning empty string")
            return ""

    async def confirm(self, prompt: str, timeout_s: float) -> str:
        """Ask for a short spoken confirmation and return the transcript."""
        await self.speak(prompt)
        audio = await self._record_audio(
            max_record_s=max(0.5, float(timeout_s)),
            silence_duration_s=min(SILENCE_DURATION_S, max(0.4, float(timeout_s) / 2.0)),
        )
        if audio is None:
            return ""

        if self._stt_method == "elevenlabs":
            return await self._stt_elevenlabs(audio)
        elif self._stt_method == "whisper":
            return await self._stt_whisper(audio)
        logger.warning("No STT backend -- returning empty confirmation")
        return ""

    async def _record_audio(
        self,
        *,
        max_record_s: float = MAX_RECORD_S,
        silence_duration_s: float = SILENCE_DURATION_S,
    ) -> Optional[bytes]:
        """Record from microphone until silence detected. Returns WAV bytes."""
        try:
            import sounddevice as sd
        except ImportError:
            logger.error("sounddevice not installed")
            return None

        logger.info("Listening... (speak now)")
        frames = []
        silence_frames = 0
        max_frames = max(1, int(max_record_s * SAMPLE_RATE / 1024))
        silence_limit = max(1, int(silence_duration_s * SAMPLE_RATE / 1024))

        try:
            stream = sd.InputStream(
                samplerate=SAMPLE_RATE, channels=CHANNELS,
                dtype="int16", blocksize=1024,
            )
            stream.start()

            for _ in range(max_frames):
                data, overflowed = stream.read(1024)
                frames.append(data.tobytes())

                # RMS silence detection
                samples = struct.unpack(f"<{len(data.tobytes())//2}h", data.tobytes())
                rms = (sum(s * s for s in samples) / len(samples)) ** 0.5

                if rms < SILENCE_THRESHOLD:
                    silence_frames += 1
                else:
                    silence_frames = 0

                if silence_frames >= silence_limit and len(frames) > 10:
                    break

            stream.stop()
            stream.close()
        except Exception as exc:
            logger.error("Audio recording error: %s", exc)
            return None

        if not frames:
            return None

        # Encode as WAV
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(b"".join(frames))
        wav_bytes = buf.getvalue()
        logger.info("Recorded %.1f s of audio (%d bytes)",
                     len(b"".join(frames)) / (SAMPLE_RATE * 2), len(wav_bytes))
        return wav_bytes

    async def _stt_elevenlabs(self, wav_bytes: bytes) -> str:
        """Transcribe using ElevenLabs SDK."""
        loop = asyncio.get_event_loop()

        # Write WAV to temp file (SDK needs a file-like object)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(wav_bytes)
            tmp_path = f.name

        try:
            def _transcribe():
                with open(tmp_path, "rb") as audio_file:
                    result = self._eleven_client.speech_to_text.convert(
                        file=audio_file,
                        model_id="scribe_v1",
                    )
                return result.text if hasattr(result, 'text') else str(result)

            text = await loop.run_in_executor(None, _transcribe)
            logger.info("ElevenLabs STT: '%s'", text)
            return text
        except Exception as exc:
            logger.error("ElevenLabs STT error: %s", exc)
            return ""
        finally:
            os.unlink(tmp_path)

    async def _stt_whisper(self, wav_bytes: bytes) -> str:
        """Transcribe using local Whisper model."""
        import whisper

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(wav_bytes)
            tmp_path = f.name

        try:
            loop = asyncio.get_event_loop()
            model = whisper.load_model("base")
            result = await loop.run_in_executor(
                None, lambda: model.transcribe(tmp_path, language="en")
            )
            text = result.get("text", "").strip()
            logger.info("Whisper STT: '%s'", text)
            return text
        finally:
            os.unlink(tmp_path)

    # ------------------------------------------------------------------
    # TTS: speak text
    # ------------------------------------------------------------------

    async def speak(self, text: str) -> None:
        """Convert text to speech and play through speakers."""
        if not text:
            return

        if self._tts_method == "elevenlabs":
            await self._tts_elevenlabs(text)
        elif self._tts_method == "pyttsx3":
            await self._tts_pyttsx3(text)
        else:
            logger.info("[NO TTS] Would say: '%s'", text)

    async def _tts_elevenlabs(self, text: str) -> None:
        """Generate and play speech via ElevenLabs SDK + sounddevice."""
        import sounddevice as sd
        import numpy as np

        loop = asyncio.get_event_loop()

        try:
            def _generate():
                # Request raw PCM so we don't need ffmpeg
                audio_iter = self._eleven_client.text_to_speech.convert(
                    text=text,
                    voice_id=self._voice_id,
                    model_id="eleven_turbo_v2",
                    output_format="pcm_22050",
                )
                # Collect all chunks
                chunks = []
                for chunk in audio_iter:
                    chunks.append(chunk)
                return b"".join(chunks)

            pcm_bytes = await loop.run_in_executor(None, _generate)

            # PCM 16-bit signed LE mono at 22050 Hz
            samples = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0

            def _play():
                sd.play(samples, samplerate=22050)
                sd.wait()

            await loop.run_in_executor(None, _play)
            logger.info("ElevenLabs TTS: '%s'", text)
        except Exception as exc:
            logger.error("ElevenLabs TTS error: %s -- falling back to pyttsx3", exc)
            await self._tts_pyttsx3(text)

    async def _tts_pyttsx3(self, text: str) -> None:
        """Fallback TTS using pyttsx3 (offline)."""
        if self._pyttsx_engine is None:
            try:
                import pyttsx3
                self._pyttsx_engine = pyttsx3.init()
            except Exception:
                logger.error("pyttsx3 init failed")
                return

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._pyttsx_engine.say, text)
        await loop.run_in_executor(None, self._pyttsx_engine.runAndWait)
        logger.info("pyttsx3 TTS: '%s'", text)

    async def stop(self) -> None:
        if self._pyttsx_engine:
            self._pyttsx_engine.stop()


# ======================================================================
# Fake voice for testing
# ======================================================================

class FakeVoiceInterface:
    """No-mic, no-speaker stub for testing."""

    def __init__(self):
        self._transcript_queue: asyncio.Queue[str] = asyncio.Queue()

    async def start(self) -> None:
        logger.info("[FAKE] Voice interface started")

    async def listen(self) -> str:
        """Block until inject_transcript() is called."""
        return await self._transcript_queue.get()

    async def speak(self, text: str) -> None:
        logger.info("[FAKE] TTS: '%s'", text)

    async def confirm(self, prompt: str, timeout_s: float) -> str:
        logger.info("[FAKE] Confirm prompt: '%s' (timeout=%.1fs)", prompt, timeout_s)
        try:
            return await asyncio.wait_for(self.listen(), timeout=max(0.1, float(timeout_s)))
        except asyncio.TimeoutError:
            return ""

    def inject_transcript(self, text: str) -> None:
        """Simulate the user saying something (for testing)."""
        self._transcript_queue.put_nowait(text)

    async def stop(self) -> None:
        pass


# ======================================================================
# Factory
# ======================================================================

def create_voice_interface(fake: bool = False):
    if fake:
        return FakeVoiceInterface()
    return VoiceInterface()


# ======================================================================
# Quick test
# ======================================================================

async def _test():
    logging.basicConfig(level=logging.INFO)

    voice = create_voice_interface()
    await voice.start()

    print(f"STT method: {voice._stt_method}")
    print(f"TTS method: {voice._tts_method}")

    # Test TTS
    print("\nTesting TTS...")
    await voice.speak("Hello, I am your autonomous wheelchair.")

    # Test STT
    print("\nTesting STT (speak into your mic)...")
    text = await voice.listen()
    print(f"You said: '{text}'")

    await voice.stop()


if __name__ == "__main__":
    import asyncio
    asyncio.run(_test())
