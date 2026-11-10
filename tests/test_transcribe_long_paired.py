"""Integration checks for duration selection and resumable chunk transcription."""

import json
import subprocess
import sys
import tempfile
import threading
import unittest
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class TranscriptionHandler(BaseHTTPRequestHandler):
    requests = 0
    fail_at = None

    def do_POST(self):
        self.rfile.read(int(self.headers["Content-Length"]))
        type(self).requests += 1
        if type(self).requests == type(self).fail_at:
            self.send_response(500)
            self.end_headers()
            self.wfile.write(b"temporary error")
            return
        # The second successful chunk can contain no speech.
        number = type(self).requests
        segments = [] if number == 3 else [
            {"start": 0.1, "end": 0.2, "text": f"chunk {number}",
             "words": [{"start": 0.1, "end": 0.2, "word": "chunk"}]}
        ]
        body = json.dumps({"segments": segments}).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


class PairedTranscriptionTest(unittest.TestCase):
    def test_cutoff_resume_silence_and_timestamps(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name, seconds in (("short", 0.5), ("boundary", 1), ("long", 2.5)):
                book = root / name
                book.mkdir()
                with wave.open(str(book / "audio.wav"), "wb") as audio:
                    audio.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
                    audio.writeframes(b"\0\0" * int(seconds * 16000))

            TranscriptionHandler.requests = 0
            TranscriptionHandler.fail_at = None
            server = ThreadingHTTPServer(("127.0.0.1", 0), TranscriptionHandler)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            self.addCleanup(server.server_close)
            self.addCleanup(server.shutdown)
            url = f"http://127.0.0.1:{server.server_port}/transcribe"

            def run(script, *args):
                return subprocess.run(
                    [sys.executable, str(ROOT / "scripts" / script), "--paired-dir", str(root),
                     "--url", url, "--max-duration", "1", *args],
                    text=True, capture_output=True, timeout=30,
                )

            short = run("transcribe_paired.py")
            self.assertEqual(short.returncode, 0, short.stderr)
            self.assertEqual(TranscriptionHandler.requests, 2)
            self.assertTrue((root / "boundary" / "metadata.json").exists())
            self.assertFalse((root / "long" / "metadata.json").exists())

            TranscriptionHandler.requests = 0
            TranscriptionHandler.fail_at = 2
            first = run("transcribe_long_paired.py", "--chunk-duration", "1")
            self.assertEqual(first.returncode, 1)
            self.assertEqual(TranscriptionHandler.requests, 2)
            self.assertFalse((root / "long" / "transcription.txt").exists())

            TranscriptionHandler.fail_at = None
            resumed = run("transcribe_long_paired.py", "--chunk-duration", "1")
            self.assertEqual(resumed.returncode, 0, resumed.stderr)
            self.assertEqual(TranscriptionHandler.requests, 4)
            self.assertIn("using saved response", resumed.stdout)
            metadata = json.loads((root / "long" / "metadata.json").read_text())
            self.assertEqual(len(metadata["chunks"]), 3)
            self.assertEqual([segment["start"] for segment in metadata["segments"]], [0.1, 2.1])
            self.assertEqual(metadata["segments"][1]["words"][0]["end"], 2.2)
            self.assertEqual((root / "long" / "transcription.txt").read_text(), "chunk 1 chunk 4\n")

            rerun = run("transcribe_long_paired.py", "--chunk-duration", "1")
            self.assertEqual(rerun.returncode, 0, rerun.stderr)
            self.assertEqual(TranscriptionHandler.requests, 4)


if __name__ == "__main__":
    unittest.main()
