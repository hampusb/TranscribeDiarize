#!/usr/bin/env python3

import argparse
import whisperx
import torch
import json
from pathlib import Path
import sys
from pyannote.audio import Pipeline
import gc
import subprocess
import tempfile
import os
import pandas as pd
import numpy as np
import warnings
from contextlib import contextmanager
import config

@contextmanager
def suppress_stdout_stderr():
    """A context manager that redirects stdout and stderr to devnull"""
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    sys.stdout = open(os.devnull, 'w')
    sys.stderr = open(os.devnull, 'w')
    try:
        yield
    finally:
        sys.stdout.close()
        sys.stderr.close()
        sys.stdout = original_stdout
        sys.stderr = original_stderr

def format_timestamp(seconds):
    """Convert seconds to HH:MM:SS format"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:06.3f}"

def write_markdown(segments, output_file):
    """Write segments to markdown file, merging consecutive segments by the same speaker."""
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("# Transcription with Speaker Diarization\n\n")

        if not segments:
            return

        current_speaker = None
        current_text_parts = []
        current_start_time = None
        current_end_time = None

        for i, segment in enumerate(segments):
            speaker = segment.get('speaker', 'Unknown')
            text = segment['text'].strip()
            start_time = segment['start']
            end_time = segment['end']

            if current_speaker is None: # First segment
                current_speaker = speaker
                current_start_time = start_time
                current_end_time = end_time
                current_text_parts.append(text)
            elif speaker == current_speaker:
                # Same speaker, append text and update end time
                current_text_parts.append(text)
                current_end_time = end_time
            else:
                # Different speaker, write out the accumulated segment for the previous speaker
                if current_text_parts: # Ensure there's something to write
                    formatted_start = format_timestamp(current_start_time)
                    formatted_end = format_timestamp(current_end_time)
                    full_text = "\n".join(current_text_parts)
                    f.write(f"## {current_speaker} [{formatted_start} - {formatted_end}]\n\n")
                    f.write(f"{full_text}\n\n")
                
                # Start new accumulation for the current segment
                current_speaker = speaker
                current_start_time = start_time
                current_end_time = end_time
                current_text_parts = [text]
            
            # If it's the last segment, write out the accumulated block
            if i == len(segments) - 1:
                if current_text_parts:
                    formatted_start = format_timestamp(current_start_time)
                    formatted_end = format_timestamp(current_end_time)
                    full_text = "\n".join(current_text_parts)
                    f.write(f"## {current_speaker} [{formatted_start} - {formatted_end}]\n\n")
                    f.write(f"{full_text}\n\n")

def cleanup_model(model):
    """Properly cleanup a model and its resources"""
    # For WhisperModel, we just need to delete it
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def convert_to_wav(input_file):
    """Convert audio file to WAV format using ffmpeg"""
    # Create a temporary file for the WAV output
    temp_wav = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
    temp_wav.close()
    
    try:
        # Convert the audio file to WAV format
        with suppress_stdout_stderr(): # Suppress ffmpeg output
            subprocess.run([
                'ffmpeg', '-i', input_file,
                '-acodec', 'pcm_s16le',  # Use 16-bit PCM encoding
                '-ar', '16000',          # Set sample rate to 16kHz
                '-ac', '1',              # Convert to mono
                '-y',                    # Overwrite output file if it exists
                temp_wav.name
            ], check=True, capture_output=True) # capture_output also helps keep it quiet
        
        return temp_wav.name
    except subprocess.CalledProcessError as e:
        # We still want to see our own error message if ffmpeg fails
        original_stderr = sys.stderr
        sys.stderr = sys.__stderr__ # Temporarily restore stderr for our print
        print(f"Error converting audio file: {e.stderr.decode() if e.stderr else e.stdout.decode() if e.stdout else 'Unknown ffmpeg error'}", file=sys.stderr)
        sys.stderr = original_stderr # Re-redirect if needed or just let finally clause handle it
        if os.path.exists(temp_wav.name):
            os.unlink(temp_wav.name)
        raise

def main():
    # Suppress specific UserWarnings that might still get through if not printed to stderr
    warnings.filterwarnings('ignore', category=UserWarning)
    warnings.filterwarnings('ignore', category=FutureWarning) 

    parser = argparse.ArgumentParser(description='Transcribe audio with speaker diarization')
    parser.add_argument('audio_file', help='Path to the audio file')
    parser.add_argument('--persons', type=int, default=2, help='Number of speakers to detect')
    parser.add_argument('--output', default='output/output.md', help='Output markdown file')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size for transcription (reduce if low on GPU memory)')
    parser.add_argument('--compute-type', choices=['float16', 'float32', 'int8'], default='float32',
                      help='Compute type for model (float16 for GPU, float32 for CPU, int8 for low memory)')
    parser.add_argument('--language', default='sv', help='Language code (default: sv for Swedish)')
    args = parser.parse_args()

    # Check if audio file exists
    if not Path(args.audio_file).exists():
        print(f"Error: Audio file '{args.audio_file}' not found", file=sys.stderr)
        sys.exit(1)

    print("Converting audio to WAV format...")
    wav_file = convert_to_wav(args.audio_file)

    try:
        print("Loading KBLab Swedish model...")
        device_type = "cuda" if torch.cuda.is_available() else "cpu"
        device = torch.device(device_type)
        
        with suppress_stdout_stderr():
            # Load KBLab Swedish model
            model = whisperx.load_model(
                config.WHISPER_MODEL,
                device_type,  # Use string device type here
                compute_type=args.compute_type,
                download_root=config.MODEL_DIR
            )
        
        print("Loading audio...")
        with suppress_stdout_stderr():
            audio = whisperx.load_audio(wav_file)
        
        print("Transcribing audio...")
        with suppress_stdout_stderr():
            result = model.transcribe(
                audio,
                batch_size=args.batch_size,
                language=args.language,
                task="transcribe"
            )
        
        print("Cleaning up transcription model...")
        # cleanup_model doesn't print much, but can be wrapped if needed
        cleanup_model(model)
        
        print("Aligning transcription...")
        with suppress_stdout_stderr():
            model_a, metadata = whisperx.load_align_model(
                language_code=args.language,
                device=device,
                model_name=config.ALIGN_MODEL,
                model_dir=config.MODEL_DIR
            )
            result = whisperx.align(
                result["segments"],
                model_a,
                metadata,
                audio,
                device,
                return_char_alignments=False
            )
        
        print("Cleaning up alignment model...")
        cleanup_model(model_a)
        
        print("Performing speaker diarization...")
        with suppress_stdout_stderr():
            diarize_model = Pipeline.from_pretrained(
                config.DIARIZATION_MODEL,
                use_auth_token=config.HF_AUTH_TOKEN
            ).to(device)
            
            diarization = diarize_model(
                wav_file,  # Use the converted WAV file
                num_speakers=args.persons
            )
        
        # Convert diarization segments to a DataFrame
        diarize_list = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            diarize_list.append({
                "start": float(turn.start),
                "end": float(turn.end),
                "speaker": f"SPEAKER_{speaker}"
            })
        diarize_df = pd.DataFrame(diarize_list)
        
        # Assign speaker labels
        with suppress_stdout_stderr():
            result = whisperx.assign_word_speakers(diarize_df, result)
        
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        print(f"Writing results to {args.output}...")
        write_markdown(result["segments"], args.output)
        print("Done!")

    except Exception as e:
        # Ensure any unexpected errors are printed to the original stderr
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__
        print(f"An unexpected error occurred: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        raise # Re-raise the exception
    finally:
        # Clean up the temporary WAV file
        if 'wav_file' in locals() and os.path.exists(wav_file):
            os.unlink(wav_file)

if __name__ == "__main__":
    main() 