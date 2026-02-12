import os

# Models
WHISPER_MODEL = "KBLab/kb-whisper-large"
ALIGN_MODEL = "KBLab/wav2vec2-large-voxrex-swedish"
DIARIZATION_MODEL = "pyannote/speaker-diarization-3.1"

# Paths
MODEL_DIR = "/path/to/models"

# Authentication (override via env var: export HF_AUTH_TOKEN=your_token)
HF_AUTH_TOKEN = os.environ.get("HF_AUTH_TOKEN", "")
