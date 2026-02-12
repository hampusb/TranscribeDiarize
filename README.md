# Svensk ljudtranskribering med talardiarisering

Transkriberar svenska ljudfiler och identifierar talare med hjälp av KBLabs svenska Whisper-modell och WhisperX.

## Krav

- Python 3.8 eller senare
- CUDA-kompatibelt GPU (rekommenderas)

## Installation

1. Skapa en virtuell miljö:
```bash
python -m venv venv
source venv/bin/activate
```

2. Installera PyTorch:
```bash
# Med GPU-stöd (CUDA)
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118

# Utan GPU
pip install torch torchaudio
```

3. Installera ctranslate2:
```bash
pip install ctranslate2
```

4. Installera övriga beroenden:
```bash
pip install -r requirements.txt
```

## Konfiguration

Kopiera `config.example.py` till `config.py` och fyll i dina inställningar:

- `WHISPER_MODEL` — Whisper-modell för transkribering
- `ALIGN_MODEL` — Modell för alignment
- `DIARIZATION_MODEL` — Modell för talardiarisering
- `MODEL_DIR` — Sökväg till lokalt lagrade modeller
- `HF_AUTH_TOKEN` — HuggingFace-token (kan även sättas via miljövariabeln `HF_AUTH_TOKEN`)

## Användning

```bash
python transcribe.py ljudfil.wav --persons 3 --output output/resultat.md
```

Argument:
- `ljudfil.wav` — Sökväg till ljudfilen (obligatorisk)
- `--persons` — Antal talare (standard: 2)
- `--output` — Utdatafil (standard: `output/output.md`)
- `--language` — Språkkod (standard: `sv`)
- `--batch-size` — Batchstorlek för transkribering (standard: 16)
- `--compute-type` — Beräkningstyp: `float16`, `float32`, `int8` (standard: `float32`)

## Utdataformat

Skriptet genererar en markdownfil:

```markdown
# Transcription with Speaker Diarization

## SPEAKER_1 [00:00:00.000 - 00:00:05.000]

Transkriberad text för talare 1...

## SPEAKER_2 [00:00:05.000 - 00:00:10.000]

Transkriberad text för talare 2...
```
