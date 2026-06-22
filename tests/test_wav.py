from __future__ import annotations

import tempfile
import unittest
import wave
from pathlib import Path

from wordpipe.asr_worker import _offline_metrics, read_wav_mono_float32


class WavTests(unittest.TestCase):
    def test_reads_pcm16_mono_as_float32(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sample.wav"
            with wave.open(str(path), "wb") as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(16000)
                wav.writeframes((0).to_bytes(2, "little", signed=True))
                wav.writeframes((32767).to_bytes(2, "little", signed=True))

            samples, sample_rate = read_wav_mono_float32(path)

        self.assertEqual(sample_rate, 16000)
        self.assertEqual(len(samples), 2)
        self.assertAlmostEqual(float(samples[0]), 0.0)
        self.assertAlmostEqual(float(samples[1]), 32767 / 32768)

    def test_offline_metrics_include_rtf(self) -> None:
        metrics = _offline_metrics(0.0, 2.0, 1.0, 3)

        self.assertEqual(metrics["real_time_factor"], 0.5)
        self.assertEqual(metrics["decode_calls"], 3)
        self.assertEqual(metrics["last_rms"], 0.0)


if __name__ == "__main__":
    unittest.main()
