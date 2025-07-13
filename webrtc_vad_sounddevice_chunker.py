#!/usr/bin/env python3
"""
Modern Real-Time WebRTC VAD Chunker (sounddevice)
=================================================

- Captures microphone audio at 16kHz, mono, 16-bit PCM
- Uses webrtcvad-wheels for robust VAD
- Saves each detected speech chunk as a WAV file in 'chunks_webrtc/'
- Prints info for each chunk
- CPU-only, no GPU required

Dependencies:
  pip install sounddevice numpy soundfile webrtcvad-wheels
"""

import sounddevice as sd
import numpy as np
import soundfile as sf
import webrtcvad_wheels as webrtcvad
import queue
import os
from collections import deque
from datetime import datetime

# --- Configuration ---
SAMPLE_RATE = 16000
CHANNELS = 1
DTYPE = 'int16'
FRAME_DURATION_MS = 30  # VAD supports 10, 20, or 30 ms
FRAME_SIZE = int(SAMPLE_RATE * FRAME_DURATION_MS / 1000)
VAD_AGGRESSIVENESS = 2  # 0 (least aggressive) to 3 (most aggressive)
SPEECH_TRIGGER_FRAMES = 7  # How many voiced frames to trigger speech
SILENCE_TRIGGER_FRAMES = 7  # How many unvoiced frames to trigger silence
PADDING_FRAMES = 10  # Frames of padding before/after speech
OUTPUT_DIR = "chunks_webrtc"

# --- Setup ---
os.makedirs(OUTPUT_DIR, exist_ok=True)
q = queue.Queue()

def audio_callback(indata, frames, time, status):
    if status:
        print(status)
    q.put(bytes(indata))

def main():
    vad = webrtcvad.Vad(VAD_AGGRESSIVENESS)
    ring_buffer = deque(maxlen=PADDING_FRAMES)
    triggered = False
    voiced_frames = []
    chunk_counter = 0

    def save_chunk(frames):
        nonlocal chunk_counter
        chunk_data = b''.join(frames)
        audio_array = np.frombuffer(chunk_data, dtype=np.int16)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(OUTPUT_DIR, f"chunk_{chunk_counter:03d}_{timestamp}.wav")
        sf.write(filename, audio_array, SAMPLE_RATE)
        print(f"💾 Saved chunk {chunk_counter}: {filename} ({len(audio_array)/SAMPLE_RATE:.2f}s)")
        chunk_counter += 1

    print("Listening (WebRTC VAD)... Press Ctrl+C to stop.")
    with sd.RawInputStream(samplerate=SAMPLE_RATE, blocksize=FRAME_SIZE, channels=CHANNELS, dtype=DTYPE, callback=audio_callback):
        try:
            while True:
                frame = q.get()
                if len(frame) != FRAME_SIZE * 2:
                    continue  # Skip incomplete frames
                is_speech = vad.is_speech(frame, SAMPLE_RATE)
                if not triggered:
                    ring_buffer.append((frame, is_speech))
                    num_voiced = len([f for f, speech in ring_buffer if speech])
                    if num_voiced > SPEECH_TRIGGER_FRAMES:
                        triggered = True
                        print("🎤 Speech started")
                        voiced_frames.extend([f for f, s in ring_buffer])
                        ring_buffer.clear()
                else:
                    voiced_frames.append(frame)
                    ring_buffer.append((frame, is_speech))
                    num_unvoiced = len([f for f, speech in ring_buffer if not speech])
                    if num_unvoiced > SILENCE_TRIGGER_FRAMES:
                        triggered = False
                        print("🔇 Speech ended")
                        save_chunk(voiced_frames)
                        voiced_frames = []
                        ring_buffer.clear()
        except KeyboardInterrupt:
            print("\nStopped by user.")
            if voiced_frames:
                save_chunk(voiced_frames)

if __name__ == "__main__":
    main() 