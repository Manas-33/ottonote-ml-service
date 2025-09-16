# OttoNote ML Service (Transcription + Diarization)

FastAPI backend providing Whisper transcription and pyannote diarization.

## Prerequisites

- Python 3.10+
- ffmpeg installed and on PATH
- CPU works; for GPU install CUDA-enabled PyTorch matching your system.

### Install ffmpeg

- macOS (brew): `brew install ffmpeg`
- Ubuntu: `sudo apt-get update && sudo apt-get install -y ffmpeg`
- Windows: Install from `https://ffmpeg.org/` and add to PATH

## Setup

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Optional environment variables:

- `WHISPER_MODEL` (default: `small`) — options: tiny, base, small, medium, large
- `HF_TOKEN` or `HUGGINGFACE_TOKEN` — required for pyannote diarization models

## Run

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8080 --reload
```

## API

- `GET /health` → `{ "status": "ok" }`

### Transcription

- `POST /v1/transcribe` — multipart form with `file`
- `POST /v1/transcribe-path` — JSON `{ "file_path": "/abs/path/audio.mp3" }`

Example:

```bash
curl -X POST -F "file=@/path/to/audio.mp3" http://localhost:8080/v1/transcribe
```

### Diarization (requires HF token)

Set your token: `export HF_TOKEN=hf_xxx`

- `POST /v1/diarize` — multipart form with `file`
- `POST /v1/diarize-path` — JSON `{ "file_path": "/abs/path/audio.mp3" }`

Example:

```bash
curl -X POST -F "file=@/path/to/audio.mp3" http://localhost:8080/v1/diarize
```

Response shape:

```json
{
  "turns": [
    { "start": 0.12, "end": 3.45, "speaker": "SPEAKER_00" },
    { "start": 3.6, "end": 7.1, "speaker": "SPEAKER_01" }
  ],
  "model": "pyannote/speaker-diarization"
}
```

## Notes

- First call will download models; it may take time.
- Ensure `ffmpeg` is installed; both Whisper and pyannote rely on it.