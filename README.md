# Swedish Audio Transcription with Diarization

This tool transcribes Swedish audio files and performs speaker diarization using the KBLab Swedish Whisper model and WhisperX.

## Requirements

- Python 3.8 or higher
- CUDA-capable GPU (recommended for faster processing)

## Installation

1. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install PyTorch first (choose the appropriate command for your system):
```bash
# For CUDA (GPU support)
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CPU only
pip install torch torchaudio
```

3. Install ctranslate2:
```bash
pip install ctranslate2
```

4. Install the remaining requirements:
```bash
pip install -r requirements.txt
```

## Usage

Basic usage:
```bash
python transcribe.py input.wav --persons 3 --output output.md
```

Arguments:
- `input.wav`: Path to the audio file (required)
- `--persons`: Number of speakers to detect (default: 2)
- `--output`: Output markdown file (default: output.md)

## Output Format

The script generates a markdown file with the following format:
```markdown
# Transcription with Speaker Diarization

## SPEAKER_1 [00:00:00.000 - 00:00:05.000]

Transcribed text for speaker 1...

## SPEAKER_2 [00:00:05.000 - 00:00:10.000]

Transcribed text for speaker 2...
```

## Notes

- The script uses the KBLab Swedish Whisper model for transcription
- Speaker diarization is performed using WhisperX
- Processing time depends on the length of the audio file and your hardware
- For best results, use clear audio with minimal background noise 