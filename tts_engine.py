import asyncio
import tempfile
import os
from pathlib import Path

VOICE = "en-US-JennyNeural"
RATE  = "+0%"
PITCH = "+0Hz"


async def _synthesize(text: str, output_path: str):
    import edge_tts
    communicate = edge_tts.Communicate(text, VOICE, rate=RATE, pitch=PITCH)
    await communicate.save(output_path)


def synthesize_speech(text: str) -> bytes:
    """
    Convert text to speech using Edge TTS (Jenny Neural voice).
    Returns MP3 bytes.
    """
    if not text or not text.strip():
        return b""
    tmp = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
    tmp.close()
    try:
        asyncio.run(_synthesize(text.strip(), tmp.name))
        with open(tmp.name, "rb") as f:
            return f.read()
    except Exception as e:
        print(f"[TTS] Edge TTS error: {e}")
        return b""
    finally:
        try:
            os.unlink(tmp.name)
        except Exception:
            pass


if __name__ == "__main__":
    audio = synthesize_speech("Hey, I'm Remi! Really glad you're here today.")
    print(f"Generated {len(audio)} bytes of audio")
    with open("test_remi.mp3", "wb") as f:
        f.write(audio)
    print("Saved test_remi.mp3 — play it to check the voice!")