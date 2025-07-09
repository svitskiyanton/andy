#!/usr/bin/env python3
"""
Test script for ElevenLabs Voice Changer setup
Run this first to verify your configuration
"""

import os
import sys
from dotenv import load_dotenv
import elevenlabs
import pyaudio

def test_environment():
    """Test environment variables"""
    print("🔍 Testing environment variables...")
    
    load_dotenv()
    
    api_key = os.getenv("ELEVENLABS_API_KEY")
    voice_id = os.getenv("VOICE_ID")
    
    if not api_key or api_key == "your_api_key_here":
        print("❌ ELEVENLABS_API_KEY not set or invalid")
        return False
    
    if not voice_id or voice_id == "your_voice_id_here":
        print("❌ VOICE_ID not set or invalid")
        return False
    
    print(f"✅ API Key: {api_key[:10]}...")
    print(f"✅ Voice ID: {voice_id}")
    return True

def test_elevenlabs_connection(api_key):
    """Test ElevenLabs API connection"""
    print("\n🔍 Testing ElevenLabs API connection...")
    
    try:
        # Set API key
        elevenlabs.set_api_key(api_key)
        
        # Test by getting available voices
        voices = elevenlabs.voices()
        print(f"✅ Connected to ElevenLabs API")
        print(f"✅ Found {len(voices)} available voices")
        
        return True
        
    except Exception as e:
        print(f"❌ ElevenLabs API connection failed: {e}")
        return False

def test_voice_id(api_key, voice_id):
    """Test if the voice ID is valid"""
    print(f"\n🔍 Testing voice ID: {voice_id}")
    
    try:
        # Set API key
        elevenlabs.set_api_key(api_key)
        
        # Try to get the specific voice
        voices = elevenlabs.voices()
        voice_found = False
        
        for voice in voices:
            if voice.voice_id == voice_id:
                print(f"✅ Voice found: {voice.name}")
                print(f"✅ Voice description: {voice.description}")
                voice_found = True
                break
        
        if not voice_found:
            print(f"❌ Voice ID {voice_id} not found in your voices")
            return False
            
        return True
        
    except Exception as e:
        print(f"❌ Voice ID test failed: {e}")
        return False

def test_audio_devices():
    """Test audio input and output devices"""
    print("\n🔍 Testing audio devices...")
    
    try:
        p = pyaudio.PyAudio()
        
        # Test input devices
        input_device = p.get_default_input_device_info()
        print(f"✅ Default input: {input_device['name']}")
        
        # Test output devices
        output_device = p.get_default_output_device_info()
        print(f"✅ Default output: {output_device['name']}")
        
        # List all devices
        print(f"\n📋 Available devices:")
        for i in range(p.get_device_count()):
            device_info = p.get_device_info_by_index(i)
            device_type = "Input" if device_info['maxInputChannels'] > 0 else "Output"
            print(f"   {i}: {device_info['name']} ({device_type})")
        
        p.terminate()
        return True
        
    except Exception as e:
        print(f"❌ Audio device test failed: {e}")
        return False

def test_speech_to_speech_api(api_key, voice_id):
    """Test the Speech-to-Speech API endpoint"""
    print(f"\n🔍 Testing Speech-to-Speech API...")
    
    try:
        import requests
        import tempfile
        import wave
        
        # Create a temporary WAV file with proper audio format
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
            # Create a simple WAV file (1 second of silence at 16kHz, 16-bit, mono)
            with wave.open(temp_file.name, 'wb') as wav_file:
                wav_file.setnchannels(1)  # Mono
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(16000)  # 16kHz
                wav_file.writeframes(b'\x00' * 32000)  # 1 second of silence
            
            # Read the file for sending
            with open(temp_file.name, 'rb') as audio_file:
                audio_data = audio_file.read()
        
        url = f"https://api.elevenlabs.io/v1/speech-to-speech/{voice_id}/stream"
        
        headers = {
            "xi-api-key": api_key
        }
        
        params = {
            "optimize_streaming_latency": 2,
            "output_format": "pcm_16000",
            "file_format": "pcm_s16le_16"
        }
        
        # Send as multipart form data
        files = {
            'audio': ('test_audio.wav', audio_data, 'audio/wav')
        }
        
        print("   Sending test request...")
        response = requests.post(
            url,
            headers=headers,
            params=params,
            files=files,
            timeout=10
        )
        
        # Clean up temporary file
        import os
        os.unlink(temp_file.name)
        
        if response.status_code == 200:
            print("✅ Speech-to-Speech API test successful")
            return True
        else:
            print(f"❌ Speech-to-Speech API test failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Speech-to-Speech API test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 ElevenLabs Voice Changer Test Suite")
    print("=" * 50)
    
    tests = [
        ("Environment Variables", test_environment),
        ("Audio Devices", test_audio_devices)
    ]
    
    # Get API key for API tests
    load_dotenv()
    api_key = os.getenv("ELEVENLABS_API_KEY")
    voice_id = os.getenv("VOICE_ID")
    
    if api_key and api_key != "your_api_key_here":
        tests.extend([
            ("ElevenLabs API Connection", lambda: test_elevenlabs_connection(api_key)),
            ("Voice ID Validation", lambda: test_voice_id(api_key, voice_id)),
            ("Speech-to-Speech API", lambda: test_speech_to_speech_api(api_key, voice_id))
        ])
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name} PASSED")
            else:
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            print(f"❌ {test_name} ERROR: {e}")
    
    print(f"\n{'='*50}")
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! You're ready to run the voice changer.")
        print("   Run: python real_time_voice_changer.py")
    else:
        print("⚠️  Some tests failed. Please fix the issues before running the voice changer.")
        print("   Check the README.md for troubleshooting tips.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 