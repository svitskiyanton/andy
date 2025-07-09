#!/usr/bin/env python3
"""
Script to verify saved MP3 files from text streaming
"""

import os
import sys
from pydub import AudioSegment
import json

def verify_mp3_file(file_path):
    """Verify and analyze an MP3 file"""
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return False
    
    try:
        # Load the audio file
        audio = AudioSegment.from_file(file_path, format="mp3")
        
        # Get file info
        file_size = os.path.getsize(file_path)
        duration = len(audio) / 1000.0  # Convert to seconds
        
        print("🔍 MP3 File Verification")
        print("=" * 50)
        print(f"📁 File: {file_path}")
        print(f"📊 File size: {file_size:,} bytes ({file_size / 1024 / 1024:.2f} MB)")
        print(f"⏱️  Duration: {duration:.2f} seconds")
        print(f"🎵 Sample rate: {audio.frame_rate} Hz")
        print(f"🔊 Channels: {audio.channels}")
        print(f"📈 Max amplitude: {audio.max_possible_amplitude}")
        print(f"🔊 Average amplitude: {audio.dBFS:.1f} dB")
        
        # Check for silence or very low volume
        if audio.dBFS < -50:
            print("⚠️  WARNING: Audio appears to be very quiet or silent")
        elif audio.dBFS < -30:
            print("⚠️  WARNING: Audio is quite quiet")
        else:
            print("✅ Audio volume appears normal")
        
        # Check for clipping
        if audio.max_possible_amplitude > 0.95:
            print("⚠️  WARNING: Audio may be clipping")
        else:
            print("✅ No clipping detected")
        
        # Calculate bitrate
        bitrate = (file_size * 8) / duration  # bits per second
        print(f"📊 Calculated bitrate: {bitrate / 1000:.0f} kbps")
        
        print("=" * 50)
        return True
        
    except Exception as e:
        print(f"❌ Error analyzing MP3 file: {e}")
        return False

def list_mp3_files():
    """List all MP3 files in current directory"""
    mp3_files = [f for f in os.listdir('.') if f.endswith('.mp3')]
    
    if not mp3_files:
        print("📁 No MP3 files found in current directory")
        return []
    
    print("📁 Found MP3 files:")
    for i, file in enumerate(mp3_files, 1):
        size = os.path.getsize(file)
        print(f"  {i}. {file} ({size:,} bytes)")
    
    return mp3_files

def main():
    """Main function"""
    if len(sys.argv) > 1:
        # Verify specific file
        file_path = sys.argv[1]
        verify_mp3_file(file_path)
    else:
        # List and verify all MP3 files
        mp3_files = list_mp3_files()
        
        if mp3_files:
            print("\n🔍 Verifying all MP3 files...")
            for file in mp3_files:
                print(f"\n📋 Analyzing: {file}")
                verify_mp3_file(file)
                print()

if __name__ == "__main__":
    main() 