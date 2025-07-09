#!/usr/bin/env python3
"""
Simple Audio Test - Test different audio playback methods
"""

import subprocess
import tempfile
import os
import platform

def test_sox_mp3():
    """Test SoX with a simple MP3 file"""
    print("🎵 Testing SoX MP3 playback...")
    
    # Create a simple test MP3 (sine wave)
    test_mp3 = os.path.join(tempfile.gettempdir(), "test_sine.mp3")
    
    # Generate a test MP3 using FFmpeg
    try:
        ffmpeg_cmd = [
            'ffmpeg',
            '-hide_banner',
            '-loglevel', 'error',
            '-f', 'lavfi',
            '-i', 'sine=frequency=440:duration=3',
            '-acodec', 'mp3',
            '-ab', '128k',
            test_mp3
        ]
        
        result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("✅ Test MP3 created successfully")
            
            # Try to play with SoX
            sox_cmd = ['sox', test_mp3, '-d']
            print("🔊 Playing test MP3 with SoX...")
            result = subprocess.run(sox_cmd, capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                print("✅ SoX MP3 playback successful")
                return True
            else:
                print(f"❌ SoX MP3 playback failed: {result.stderr}")
                return False
        else:
            print(f"❌ Failed to create test MP3: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Error testing SoX: {e}")
        return False

def test_ffmpeg_direct():
    """Test FFmpeg direct audio output"""
    print("🎵 Testing FFmpeg direct audio output...")
    
    test_mp3 = os.path.join(tempfile.gettempdir(), "test_sine.mp3")
    
    try:
        # Generate test MP3 if it doesn't exist
        if not os.path.exists(test_mp3):
            ffmpeg_cmd = [
                'ffmpeg',
                '-hide_banner',
                '-loglevel', 'error',
                '-f', 'lavfi',
                '-i', 'sine=frequency=440:duration=3',
                '-acodec', 'mp3',
                '-ab', '128k',
                test_mp3
            ]
            subprocess.run(ffmpeg_cmd, capture_output=True, text=True, timeout=10)
        
        # Try FFmpeg direct output
        if platform.system() == "Windows":
            # Use DirectShow on Windows
            ffmpeg_play_cmd = [
                'ffmpeg',
                '-hide_banner',
                '-loglevel', 'error',
                '-i', test_mp3,
                '-f', 'wav',
                '-acodec', 'pcm_s16le',
                '-ar', '44100',
                '-ac', '1',
                '-f', 'wav',
                'pipe:1'
            ]
        else:
            # Use ALSA on Linux
            ffmpeg_play_cmd = [
                'ffmpeg',
                '-hide_banner',
                '-loglevel', 'error',
                '-i', test_mp3,
                '-f', 'alsa',
                'default'
            ]
        
        print("🔊 Playing test MP3 with FFmpeg...")
        result = subprocess.run(ffmpeg_play_cmd, capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            print("✅ FFmpeg direct playback successful")
            return True
        else:
            print(f"❌ FFmpeg direct playback failed: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Error testing FFmpeg: {e}")
        return False

def test_mpv():
    """Test MPV player"""
    print("🎵 Testing MPV player...")
    
    test_mp3 = os.path.join(tempfile.gettempdir(), "test_sine.mp3")
    
    try:
        # Generate test MP3 if it doesn't exist
        if not os.path.exists(test_mp3):
            ffmpeg_cmd = [
                'ffmpeg',
                '-hide_banner',
                '-loglevel', 'error',
                '-f', 'lavfi',
                '-i', 'sine=frequency=440:duration=3',
                '-acodec', 'mp3',
                '-ab', '128k',
                test_mp3
            ]
            subprocess.run(ffmpeg_cmd, capture_output=True, text=True, timeout=10)
        
        # Try MPV
        mpv_cmd = [
            'mpv',
            '--no-video',
            '--no-terminal',
            test_mp3
        ]
        
        print("🔊 Playing test MP3 with MPV...")
        result = subprocess.run(mpv_cmd, capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            print("✅ MPV playback successful")
            return True
        else:
            print(f"❌ MPV playback failed: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Error testing MPV: {e}")
        return False

def main():
    """Run all audio tests"""
    print("🎵 Simple Audio Test")
    print("=" * 40)
    print(f"Platform: {platform.system()}")
    print()
    
    # Test different audio players
    tests = [
        ("SoX MP3", test_sox_mp3),
        ("FFmpeg Direct", test_ffmpeg_direct),
        ("MPV", test_mpv),
    ]
    
    results = {}
    for name, test_func in tests:
        print(f"Testing {name}...")
        results[name] = test_func()
        print()
    
    # Summary
    print("=" * 40)
    print("TEST RESULTS:")
    for name, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{name}: {status}")
    
    # Recommendations
    print("\nRECOMMENDATIONS:")
    if results["SoX MP3"]:
        print("✅ Use SoX for MP3 streaming")
    elif results["FFmpeg Direct"]:
        print("✅ Use FFmpeg direct output")
    elif results["MPV"]:
        print("✅ Use MPV for audio playback")
    else:
        print("❌ No audio playback method working")
        print("   Please check your audio drivers and player installations")

if __name__ == "__main__":
    main() 