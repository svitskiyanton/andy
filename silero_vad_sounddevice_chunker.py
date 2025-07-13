#!/usr/bin/env python3
"""
Real-Time Silero VAD Chunker (sounddevice)
=========================================

- Captures microphone audio at 16kHz, mono, 16-bit PCM
- Uses Silero VAD (PyTorch, CPU) for robust speech detection
- Saves each detected speech chunk as a WAV file in 'chunks_silero/'
- Prints info for each chunk
- CPU-only, no GPU required

Dependencies:
  pip install sounddevice numpy soundfile torch torchaudio silero-vad
"""

import sounddevice as sd
import numpy as np
import soundfile as sf
import torch
import queue
import os
from collections import deque
from datetime import datetime

# --- Configuration ---
SAMPLE_RATE = 16000
CHANNELS = 1
DTYPE = 'int16'
WINDOW_SECONDS = 0.5  # How often to run VAD analysis (in seconds)
BUFFER_SECONDS = 10.0  # Size of the audio buffer to analyze (in seconds)
WINDOW_SIZE_SAMPLES = int(SAMPLE_RATE * WINDOW_SECONDS)
BUFFER_SAMPLES = int(SAMPLE_RATE * BUFFER_SECONDS)
OUTPUT_DIR = "chunks_silero"

# --- Setup ---
os.makedirs(OUTPUT_DIR, exist_ok=True)
q = queue.Queue()

print("Loading Silero VAD model...")
model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', force_reload=False)
(get_speech_timestamps, _, _, VADIterator, collect_chunks) = utils
print("Model loaded.")

def audio_callback(indata, frames, time, status):
    if status:
        print(status)
    q.put(indata.copy())

def main():
    audio_buffer = deque(maxlen=BUFFER_SAMPLES)
    chunk_counter = 0
    print("Listening (Silero VAD)... Press Ctrl+C to stop.")
    with sd.InputStream(samplerate=SAMPLE_RATE, blocksize=WINDOW_SIZE_SAMPLES, channels=CHANNELS, dtype=DTYPE, callback=audio_callback):
        try:
            while True:
                # Wait for enough data
                if len(audio_buffer) < WINDOW_SIZE_SAMPLES:
                    indata = q.get()
                    audio_buffer.extend(indata.flatten())
                    continue
                # Analyze the buffer
                current_window = np.array(audio_buffer, dtype=np.int16)
                audio_float32 = current_window.astype(np.float32) / 32768.0
                audio_tensor = torch.from_numpy(audio_float32)
                speech_timestamps = get_speech_timestamps(audio_tensor, model, sampling_rate=SAMPLE_RATE)
                if speech_timestamps:
                    for seg in speech_timestamps:
                        start = seg['start']
                        end = seg['end']
                        if end > len(current_window):
                            continue
                        speech_segment = current_window[start:end]
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        filename = os.path.join(OUTPUT_DIR, f"chunk_{chunk_counter:03d}_{timestamp}.wav")
                        sf.write(filename, speech_segment, SAMPLE_RATE)
                        print(f"💾 Saved chunk {chunk_counter}: {filename} ({len(speech_segment)/SAMPLE_RATE:.2f}s)")
                        chunk_counter += 1
                    # Remove processed samples
                    last_end = speech_timestamps[-1]['end']
                    for _ in range(last_end):
                        if audio_buffer:
                            audio_buffer.popleft()
                else:
                    # Remove a window's worth to keep buffer moving
                    for _ in range(WINDOW_SIZE_SAMPLES):
                        if audio_buffer:
                            audio_buffer.popleft()
        except KeyboardInterrupt:
            print("\nStopped by user.")

if __name__ == "__main__":
    main() 