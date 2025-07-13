#!/usr/bin/env python3
"""
Create Test Audio File for Pro STS Batch Test
Generates a simple MP3 file with speech for testing
"""

import os
import time
from pydub import AudioSegment
from pydub.generators import Sine
import numpy as np

def create_test_audio():
    """Create a test MP3 file with speech-like audio"""
    
    # Create a simple test audio file
    print("🎵 Creating test audio file...")
    
    # Generate a simple tone (simulating speech)
    sample_rate = 44100
    duration_ms = 5000  # 5 seconds
    
    # Create a sine wave that varies in frequency (simulating speech)
    audio = AudioSegment.silent(duration=duration_ms)
    
    # Add some variation to simulate speech
    for i in range(0, duration_ms, 100):  # Every 100ms
        freq = 200 + (i % 1000)  # Varying frequency
        tone = Sine(freq).to_audio_segment(duration=100)
        audio = audio.overlay(tone, position=i)
    
    # Add some silence at the beginning and end
    silence = AudioSegment.silent(duration=500)
    audio = silence + audio + silence
    
    # Export as MP3
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"test_audio_{timestamp}.mp3"
    
    audio.export(filename, format="mp3", bitrate="128k")
    
    print(f"✅ Test audio file created: {filename}")
    print(f"📊 Duration: {len(audio)}ms ({len(audio)/1000:.1f}s)")
    print(f"📁 File size: {os.path.getsize(filename)} bytes")
    
    return filename

if __name__ == "__main__":
    test_file = create_test_audio()
    print(f"\n🎤 You can now use this file for testing:")
    print(f"   python test_pro_sts_batch.py {test_file}") 