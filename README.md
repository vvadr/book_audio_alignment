# Extract paired-book PDFs

To create `data/paired/<book>/book.txt` for every book, run:

```bash
python scripts/pdf_to_txt_paired.py
```

The script examines PDFs in each book's `text/` directory and book directory.
It selects one edition, preferring the book over lesson plans or teacher guides
and an edition with usable text over a scan. Duplicate PDFs are processed once.
It extracts embedded text and uses Tesseract at 300 DPI for pages with missing
or visibly broken text. Original Latin/Cyrillic spelling and line breaks are
preserved, with repeated margins, page numbers and line-end word splits cleaned.

PyMuPDF is a project dependency. OCR also needs Tesseract with Uzbek Latin and
Cyrillic language packs (Ubuntu: `tesseract-ocr`, `tesseract-ocr-uzb`,
`tesseract-ocr-uzb-cyrl`). The script also detects a local Tesseract installation
under `.local/ocr/usr/`, or accepts `--tesseract` and `--tessdata` paths.

Existing nonempty `book.txt` files are skipped; `--force` replaces them. OCR page
results are cached in `data/paired/.pdf_text_cache/`, so interrupted runs resume.
Use `--workers 16` to run more OCR pages concurrently, `--book FOLDER_NAME` to
select a book, `--force-ocr` to disregard embedded text, or `--analyze-only` to
inspect source choices without conversion. `--paired-dir`, `--language` and
`--dpi` are also configurable. `--force-ocr` still uses matching OCR checkpoints.

Each converted book gets `book_extraction.json` with its source, page counts and
warnings. `data/paired/pdf_extraction_report.json` contains the collection report,
including alternative PDFs, missing full textbooks and title/grade/part
mismatches. Only the supplied PDFs are converted; incorrect or incomplete source
pairings require fixing the input files. OCR can contain recognition errors.
Failures are reported per book, processing continues, and the script exits
nonzero if any book failed.

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

## Transcribe long paired books

The paired script processes recordings up to **600 seconds (10 minutes)** and
skips longer recordings. This is a local selection limit; previously the script
had no duration cutoff. To process the longer recordings, run:

```bash
python scripts/transcribe_long_paired.py
```

The long-audio script splits each recording into 300-second chunks, transcribes
them in order, and writes the same `transcription.txt` and `metadata.json`
outputs in the book directory. Segment and word timestamps in `segments` are
relative to the original recording; `chunks` retains each original API response
and its offset. Chunk boundaries can affect recognition at the split. Successful
chunk responses are saved under each book's `.transcribe_long_chunks` directory,
so a rerun resumes after an interruption. Changing the audio, endpoint, or chunk
size starts a new set of checkpoints. Once outputs exist, both scripts skip the
book; `--force` replaces outputs and, for long audio, re-transcribes every chunk.

Both scripts require `ffprobe`; the long-audio script also requires `ffmpeg`.
Use `--max-duration SECONDS` with the **same value on both scripts** to change
the cutoff, and `--chunk-duration SECONDS` to change the long script's chunk
size. Both scripts also support `--paired-dir`, `--url`, and `--timeout`. Each
continues to the next book after an error and exits nonzero if any book failed.
