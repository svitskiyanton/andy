#!/usr/bin/env python3
"""
Check ElevenLabs API Limits
Test different endpoints and understand free tier restrictions
"""

import os
import requests
import json
import time
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def check_api_limits():
    """Check ElevenLabs API limits and restrictions"""
    api_key = os.getenv("ELEVENLABS_API_KEY")
    if not api_key:
        print("❌ ELEVENLABS_API_KEY not found")
        return
    
    print("🔍 Checking ElevenLabs API Limits...")
    print("=" * 50)
    
    headers = {
        "xi-api-key": api_key,
        "Content-Type": "application/json"
    }
    
    # Test 1: Get user info and subscription
    print("\n1. Checking user subscription...")
    try:
        response = requests.get("https://api.elevenlabs.io/v1/user", headers=headers)
        if response.status_code == 200:
            user_data = response.json()
            print(f"✅ User: {user_data.get('first_name', 'Unknown')}")
            print(f"✅ Subscription: {user_data.get('subscription', {}).get('tier', 'Unknown')}")
            print(f"✅ Character count: {user_data.get('subscription', {}).get('character_count', 0)}")
            print(f"✅ Character limit: {user_data.get('subscription', {}).get('character_limit', 0)}")
        else:
            print(f"❌ Failed to get user info: {response.status_code}")
    except Exception as e:
        print(f"❌ Error getting user info: {e}")
    
    # Test 2: Get available voices
    print("\n2. Checking available voices...")
    try:
        response = requests.get("https://api.elevenlabs.io/v1/voices", headers=headers)
        if response.status_code == 200:
            voices = response.json()
            print(f"✅ Available voices: {len(voices.get('voices', []))}")
            for voice in voices.get('voices', [])[:3]:  # Show first 3
                print(f"   - {voice.get('name', 'Unknown')} (ID: {voice.get('voice_id', 'Unknown')})")
        else:
            print(f"❌ Failed to get voices: {response.status_code}")
    except Exception as e:
        print(f"❌ Error getting voices: {e}")
    
    # Test 3: Test Text-to-Speech REST API
    print("\n3. Testing Text-to-Speech REST API...")
    voice_id = os.getenv("VOICE_ID", "21m00Tcm4TlvDq8ikWAM")
    
    test_text = "Привет, это тест."
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
    
    data = {
        "text": test_text,
        "model_id": "eleven_multilingual_v2",
        "voice_settings": {
            "stability": 0.5,
            "similarity_boost": 0.8
        }
    }
    
    try:
        start_time = time.time()
        response = requests.post(url, json=data, headers=headers, timeout=30)
        end_time = time.time()
        
        if response.status_code == 200:
            print(f"✅ REST API TTS successful (latency: {end_time - start_time:.3f}s)")
            print(f"✅ Audio size: {len(response.content)} bytes")
        else:
            print(f"❌ REST API TTS failed: {response.status_code}")
            print(f"❌ Error: {response.text}")
    except Exception as e:
        print(f"❌ REST API TTS error: {e}")
    
    # Test 4: Check rate limits
    print("\n4. Testing rate limits...")
    try:
        # Make multiple requests quickly
        for i in range(3):
            start_time = time.time()
            response = requests.post(url, json=data, headers=headers, timeout=10)
            end_time = time.time()
            
            if response.status_code == 200:
                print(f"✅ Request {i+1}: Success (latency: {end_time - start_time:.3f}s)")
            elif response.status_code == 429:
                print(f"⚠️ Request {i+1}: Rate limited (429)")
                break
            else:
                print(f"❌ Request {i+1}: Failed ({response.status_code})")
            
            time.sleep(0.5)  # Small delay between requests
    except Exception as e:
        print(f"❌ Rate limit test error: {e}")
    
    # Test 5: Check WebSocket endpoint
    print("\n5. Checking WebSocket endpoint...")
    try:
        # Test if WebSocket endpoint is accessible
        ws_url = f"wss://api.elevenlabs.io/v1/text-to-speech/{voice_id}/stream-input?model_id=eleven_multilingual_v2"
        print(f"✅ WebSocket URL: {ws_url}")
        print("   (WebSocket testing requires async connection)")
    except Exception as e:
        print(f"❌ WebSocket URL error: {e}")
    
    print("\n" + "=" * 50)
    print("📋 ELEVENLABS FREE TIER LIMITATIONS:")
    print("=" * 50)
    print("• 10,000 characters per month")
    print("• 5 voices available")
    print("• Rate limiting on requests")
    print("• WebSocket may have connection limits")
    print("• No streaming optimization on free tier")
    print("• Limited concurrent connections")
    print("\n💡 RECOMMENDATIONS:")
    print("• Use REST API for reliability")
    print("• Implement rate limiting in your app")
    print("• Monitor character usage")
    print("• Consider upgrading for better performance")

if __name__ == "__main__":
    check_api_limits() 