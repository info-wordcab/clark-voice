"""Audio utilities for recording and format conversion."""

import io
import wave
from typing import Optional


def create_wav_bytes(
    audio_data: bytes,
    sample_rate: int,
    num_channels: int,
    sample_width: int = 2,
) -> bytes:
    """Convert raw PCM audio data to WAV format.

    Args:
        audio_data: Raw PCM audio bytes (16-bit signed integers)
        sample_rate: Sample rate in Hz (e.g., 8000, 16000)
        num_channels: Number of channels (1 for mono, 2 for stereo)
        sample_width: Bytes per sample (2 for 16-bit audio)

    Returns:
        WAV file as bytes with proper header
    """
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(num_channels)
        wav_file.setsampwidth(sample_width)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_data)
    return buffer.getvalue()


def save_wav_file(
    filepath: str,
    audio_data: bytes,
    sample_rate: int,
    num_channels: int,
    sample_width: int = 2,
) -> None:
    """Save raw PCM audio data to a WAV file.

    Args:
        filepath: Path to save the WAV file
        audio_data: Raw PCM audio bytes
        sample_rate: Sample rate in Hz
        num_channels: Number of channels
        sample_width: Bytes per sample
    """
    with wave.open(filepath, "wb") as wav_file:
        wav_file.setnchannels(num_channels)
        wav_file.setsampwidth(sample_width)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_data)


def get_audio_duration(audio_data: bytes, sample_rate: int, num_channels: int) -> float:
    """Calculate duration of audio data in seconds.

    Args:
        audio_data: Raw PCM audio bytes (16-bit)
        sample_rate: Sample rate in Hz
        num_channels: Number of channels

    Returns:
        Duration in seconds
    """
    # 16-bit audio = 2 bytes per sample
    bytes_per_sample = 2
    total_samples = len(audio_data) // (bytes_per_sample * num_channels)
    return total_samples / sample_rate
