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
