# Setup (macOS and the DGX Spark)

One conda environment, `book-audio-alignment`, defined in `environment.yml`,
works on both machines with the same package versions:

| Machine | GPU used | PyTorch build |
|---|---|---|
| Mac (Apple silicon) | Apple GPU (MPS) | `torch 2.14.1` |
| DGX Spark, Linux aarch64 | NVIDIA GB10 (CUDA) | `torch 2.14.1+cu130` |

Python (3.12.16) and FFmpeg (9.0.2) also come from the environment, so you
don't need a system FFmpeg. Install [Miniforge](https://conda-forge.org/download/),
then from the project root run:

```bash
conda env create -f environment.yml
conda activate book-audio-alignment
python -m ipykernel install --user --name book-audio-alignment \
  --display-name book-audio-alignment \
  --env PATH "$CONDA_PREFIX/bin:\${PATH}" --env PYTORCH_ENABLE_MPS_FALLBACK 1
python scripts/check_env.py
```

`check_env.py` should end with `RESULT OK` and show `cuda` on the Spark and
`mps` on the Mac. In your notebook editor, select the `book-audio-alignment`
kernel. The notebooks use paths relative to `src/notebooks`, so use that
directory as the notebook working directory.

Code that uses a model picks the GPU automatically, so the same code runs on
both machines:

```python
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
```

Don't hard-code `device="cuda"`, because that fails on the Mac.

To change dependencies, edit `environment.yml`, then run
`conda env update -f environment.yml --prune` on both machines. The model
notebooks download model files on their first run if they aren't cached.

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
