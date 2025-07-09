#!/usr/bin/env python3
"""
Simple audio player for testing debug output
"""

import pyaudio
import wave
import sys
import os

def play_wav_file(filename):
    """Play a WAV file"""
    if not os.path.exists(filename):
        print(f"❌ File not found: {filename}")
        return
    
    try:
        # Open the WAV file
        wf = wave.open(filename, 'rb')
        
        # Initialize PyAudio
        p = pyaudio.PyAudio()
        
        # Open stream
        stream = p.open(
            format=p.get_format_from_width(wf.getsampwidth()),
            channels=wf.getnchannels(),
            rate=wf.getframerate(),
            output=True
        )
        
        print(f"🔊 Playing: {filename}")
        print(f"📊 Duration: {wf.getnframes() / wf.getframerate():.1f} seconds")
        print(f"📊 Sample Rate: {wf.getframerate()} Hz")
        print(f"📊 Channels: {wf.getnchannels()}")
        print("🎵 Press Ctrl+C to stop")
        
        # Read data in chunks and play
        chunk_size = 1024
        data = wf.readframes(chunk_size)
        
        while data:
            stream.write(data)
            data = wf.readframes(chunk_size)
        
        # Clean up
        stream.stop_stream()
        stream.close()
        p.terminate()
        wf.close()
        
        print("✅ Playback complete")
        
    except Exception as e:
        print(f"❌ Error playing audio: {e}")

def main():
    """Main function"""
    filename = "debug_output.wav"
    
    if len(sys.argv) > 1:
        filename = sys.argv[1]
    
    play_wav_file(filename)

if __name__ == "__main__":
    main() 