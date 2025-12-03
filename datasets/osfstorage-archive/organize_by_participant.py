#!/usr/bin/env python3

import os
import shutil
from pathlib import Path

def organize_transcripts():
    # Source directory containing the story folders
    source_dir = Path("data/recall_transcripts")
    
    # Target directory for participant folders
    target_dir = Path("participants")
    target_dir.mkdir(exist_ok=True)
    
    # Story types to process
    story_types = ["pieman", "oregontrail", "eyespy", "baseball"]
    
    for story_type in story_types:
        story_dir = source_dir / story_type
        if not story_dir.exists():
            continue
            
        # Process all transcript files in this story directory
        for transcript_file in story_dir.glob("*.txt"):
            # Extract participant ID from filename (e.g., P117_baseball.txt -> P117)
            participant_id = transcript_file.stem.split("_")[0]
            
            # Create participant directory if it doesn't exist
            participant_dir = target_dir / participant_id
            participant_dir.mkdir(exist_ok=True)
            
            # Copy the transcript to the participant's folder
            target_file = participant_dir / transcript_file.name
            shutil.copy2(transcript_file, target_file)
            print(f"Copied {transcript_file} to {target_file}")

if __name__ == "__main__":
    organize_transcripts()
    print("Organization complete!")
