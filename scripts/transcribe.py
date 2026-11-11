"""Send an audio file to the transcription API (CLI and reusable functions)."""

import argparse
import http.client
import json
import mimetypes
import os
import sys
import uuid
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlsplit

from segments import join_segments

API_URL = "http://192.168.1.71:8194/transcribe"
DEFAULT_AUDIO = Path(__file__).resolve().parent.parent / "data" / "test" / "test_audio.mp3"


def transcribe(audio, url=None, *, timeout=600):
    """Upload an audio file and return the API's original response bytes."""
    url = url or os.environ.get("TRANSCRIBE_URL") or API_URL
    audio = Path(audio).expanduser()
    parts = urlsplit(url)

    if (
        parts.scheme not in ("http", "https")
        or not parts.hostname
        or parts.username
        or parts.password
        or parts.fragment
    ):
        raise ValueError("URL must be an HTTP(S) endpoint without credentials or a fragment")

    query = [
        (key, value)
        for key, value in parse_qsl(parts.query, keep_blank_values=True)
        if key not in ("language", "min_probability", "min_probabilty")
    ]
    query.extend((("language", "uz"), ("min_probability", "0")))
    target = (parts.path or "/") + "?" + urlencode(query)

    boundary = uuid.uuid4().hex
    content_type = mimetypes.guess_type(audio.name)[0] or "application/octet-stream"
    extension = audio.suffix if audio.suffix[1:].isalnum() else ""
    prefix = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="audio{extension}"\r\n'
        f"Content-Type: {content_type}\r\n\r\n"
    ).encode()
    suffix = f"\r\n--{boundary}--\r\n".encode()

    connection_class = (
        http.client.HTTPSConnection if parts.scheme == "https"
        else http.client.HTTPConnection
    )

    with audio.open("rb") as source:
        size = audio.stat().st_size
        connection = connection_class(parts.hostname, parts.port, timeout=timeout)
        try:
            connection.putrequest("POST", target)
            connection.putheader("Content-Type", f"multipart/form-data; boundary={boundary}")
            connection.putheader("Content-Length", str(len(prefix) + size + len(suffix)))
            connection.endheaders()
            connection.send(prefix)
            while chunk := source.read(1024 * 1024):
                connection.send(chunk)
            connection.send(suffix)

            response = connection.getresponse()
            body = response.read()
            if not 200 <= response.status < 300:
                message = body.decode("utf-8", errors="replace")
                raise RuntimeError(
                    f"HTTP {response.status} {response.reason}: {message}"
                )
            return body
        finally:
            connection.close()


def extract_text(response_body):
    """Extract a transcription from plain text or common JSON response shapes.

    Unknown JSON shapes are rejected rather than silently misread as words.
    """
    decoded = response_body.decode("utf-8-sig").strip()
    if not decoded:
        raise ValueError("The transcription API returned an empty response")

    try:
        payload = json.loads(decoded)
    except json.JSONDecodeError:
        return decoded

    def from_payload(value):
        if isinstance(value, str) and value.strip():
            return value.strip()
        if isinstance(value, dict):
            if isinstance(value.get("segments"), list):
                text = join_segments(value["segments"])
                if text:
                    return text
            if "text" in value:
                return from_payload(value["text"])
        return None

    text = from_payload(payload)
    if text is None:
        raise ValueError("The transcription API response has no text in a known place")
    return text


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", nargs="?", type=Path, default=DEFAULT_AUDIO,
                        help="audio file (default: data/test/test_audio.mp3)")
    parser.add_argument("url", nargs="?", default=None,
                        help="endpoint (default: TRANSCRIBE_URL, then API_URL)")
    parser.add_argument("-o", "--output", type=Path, default=None,
                        help="save the response exactly as returned")
    parser.add_argument("--timeout", type=float, default=600,
                        help="socket timeout in seconds (default 600)")
    args = parser.parse_args()

    url = args.url or os.environ.get("TRANSCRIBE_URL") or API_URL
    if not url:
        url = input("Transcription endpoint URL: ").strip()

    response = transcribe(args.audio, url, timeout=args.timeout)

    if args.output:
        args.output.write_bytes(response)
        print("Saved the response to", args.output, file=sys.stderr)

    print(response.decode("utf-8", errors="replace"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
