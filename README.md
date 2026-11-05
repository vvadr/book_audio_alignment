# Setup on macOS

Install Python 3.12 and [uv](https://docs.astral.sh/uv/), then from the project root run:

```bash
uv sync --locked
source .venv/bin/activate
```

For audio tools, install FFmpeg (`brew install ffmpeg`), which provides both
`ffmpeg` and `ffprobe`. In your notebook editor, select the project's
`.venv/bin/python` as the kernel and restart the kernel after installing packages.
Run dependent cells from the top. The notebooks use paths relative to
`src/notebooks`, so use that directory as the notebook working directory.

The lockfile is the source of truth for dependencies. `requirements.txt` is a
portable export of it and can also be installed with
`uv pip install -r requirements.txt` while `.venv` is active. Regenerate it after
dependency changes with:

```bash
uv export --locked --format requirements-txt --no-hashes --no-emit-project --output-file requirements.txt
```

The model notebooks may download model files on their first run. The second
model notebook selects CUDA, Apple Metal (MPS), or CPU according to availability.

# Transcribe audio

Set `API_URL` near the top of `scripts/transcribe.py` to your full endpoint
(including `/transcribe`), then run:

```bash
python scripts/transcribe.py
```

This uses `data/test/test_audio.mp3` and prints the response. If `API_URL` is
empty, the script asks for the endpoint. To use another audio file and save
the response:

```bash
python scripts/transcribe.py path/to/audio.mp3 -o transcription.json
```

You can also set the `TRANSCRIBE_URL` environment variable or pass the URL
after the audio path:

```bash
python scripts/transcribe.py path/to/audio.mp3 https://your-api.example/transcribe
```

The script sends the audio as
the multipart `file` field, with `language=uz` and `min_probabilty=0` as
query parameters. The parameter spelling matches the API name provided. The
response is saved exactly as returned. Use `--timeout` to change the socket
timeout in seconds (default 600).
