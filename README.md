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

## Transcribe combined paired books

To transcribe the combined recordings in `data/paired` when ready, run:

```bash
python scripts/transcribe_paired.py
```

For each book with an eligible recording, this writes `transcription.txt` and
`metadata.json` in the book directory. The metadata contains the API's JSON
response, including segment timestamps. If both `audio/combined.mp3` and
`combined.mp3` exist, the recording in `audio/` is used. If neither exists,
the script uses a single audio file in the book or its `audio/` directory,
regardless of its name. It skips books with no audio or multiple audio files
and no `combined.mp3`. Supported extensions are MP3, WAV, M4A, FLAC, OGG,
Opus, AAC and WMA. Existing outputs are left untouched; use
`--force` to replace them. Use `--paired-dir` to select another collection,
`--url` to override the API endpoint, and `--timeout` to set the socket timeout
in seconds. The script continues with other books if one fails and exits with
a nonzero status if any transcription failed.
