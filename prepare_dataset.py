import os
import random
import shutil

# Correct human source path (double folder level)
HUMAN_SOURCE = r"C:\Users\badri\Downloads\archive\cv-valid-train\cv-valid-train"
HUMAN_TARGET = r"C:\Users\badri\OneDrive\Desktop\hackathon-1\voice-detector\dataset\human"

NUM_HUMAN_FILES = 3000  # same count as AI

print("Collecting human audio files...")

# Collect all mp3 files
human_files = [
    f for f in os.listdir(HUMAN_SOURCE)
    if f.endswith(".mp3")
]

print(f"Total human files found: {len(human_files)}")

# Randomly select 3000
selected_files = random.sample(human_files, NUM_HUMAN_FILES)

print(f"Copying {len(selected_files)} human files...")

# Copy files
for filename in selected_files:
    src = os.path.join(HUMAN_SOURCE, filename)
    dst = os.path.join(HUMAN_TARGET, filename)
    shutil.copy(src, dst)

print("Human dataset prepared.")
