#!/usr/bin/env python3

import argparse
import whisperx
import torch
import json
from pathlib import Path
import sys

def format_timestamp(seconds):
    """Convert seconds to HH:MM:SS format"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:06.3f}"

def write_markdown(segments, output_file):
    """Write segments to markdown file"""
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("# Transcription with Speaker Diarization\n\n")
        
        for segment in segments:
            start_time = format_timestamp(segment['start'])
            end_time = format_timestamp(segment['end'])
            speaker = segment.get('speaker', 'Unknown')
            text = segment['text'].strip()
            
            f.write(f"## {speaker} [{start_time} - {end_time}]\n\n")
            f.write(f"{text}\n\n")

def main():
    parser = argparse.ArgumentParser(description='Transcribe audio with speaker diarization')
    parser.add_argument('audio_file', help='Path to the audio file')
    parser.add_argument('--persons', type=int, default=2, help='Number of speakers to detect')
    parser.add_argument('--output', default='output.md', help='Output markdown file')
    args = parser.parse_args()

    # Check if audio file exists
    if not Path(args.audio_file).exists():
        print(f"Error: Audio file '{args.audio_file}' not found")
        sys.exit(1)

    print("Loading KBLab Swedish model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load KBLab Swedish model
    
    model = whisperx.load_model("KBLab/kb-whisper-large", device)
    
    print("Transcribing audio...")
    # Transcribe audio
    result = model.transcribe(args.audio_file, batch_size=16)
    
    print("Aligning transcription...")
    # Align whisper output
    model_a, metadata = whisperx.load_align_model(language_code="sv", device=device)
    result = whisperx.align(result["segments"], model_a, metadata, args.audio_file, device)
    
    print("Performing speaker diarization...")
    # Perform diarization
    diarize_model = whisperx.DiarizationPipeline(use_auth_token=None, device=device)
    diarize_segments = diarize_model(args.audio_file, min_speakers=args.persons, max_speakers=args.persons)
    
    # Assign speaker labels
    result = whisperx.assign_word_speakers(diarize_segments, result)
    
    print(f"Writing results to {args.output}...")
    write_markdown(result["segments"], args.output)
    print("Done!")

if __name__ == "__main__":
    main() 