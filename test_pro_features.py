#!/usr/bin/env python3
"""
Quick Pro Features Test
Simple script to verify your ElevenLabs Pro subscription is working
"""

import os
import requests
import asyncio
import time
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_pro_subscription():
    """Test if Pro subscription is active"""
    print("🔍 Testing ElevenLabs Pro Subscription...")
    
    api_key = os.getenv("ELEVENLABS_API_KEY")
    if not api_key:
        print("❌ ELEVENLABS_API_KEY not found")
        return False
    
    headers = {
        "xi-api-key": api_key,
        "Content-Type": "application/json"
    }
    
    try:
        # Get user info
        response = requests.get("https://api.elevenlabs.io/v1/user", headers=headers)
        
        if response.status_code == 200:
            user_data = response.json()
            subscription = user_data.get('subscription', {})
            tier = subscription.get('tier', 'Unknown')
            
            print(f"✅ API Key valid")
            print(f"📊 Subscription: {tier}")
            print(f"💳 Character Count: {subscription.get('character_count', 0):,}")
            print(f"📈 Character Limit: {subscription.get('character_limit', 0):,}")
            
            # Check if Pro
            if 'pro' in tier.lower():
                print("🎉 Pro subscription confirmed!")
                return True
            else:
                print("⚠️  This appears to be a non-Pro subscription")
                return False
        else:
            print(f"❌ API test failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_pro_models():
    """Test Pro-specific models"""
    print("\n🎵 Testing Pro Models...")
    
    api_key = os.getenv("ELEVENLABS_API_KEY")
    headers = {
        "xi-api-key": api_key,
        "Content-Type": "application/json"
    }
    
    try:
        response = requests.get("https://api.elevenlabs.io/v1/models", headers=headers)
        
        if response.status_code == 200:
            models = response.json()
            
            # Check for Turbo models (Pro feature)
            turbo_models = [m for m in models if 'turbo' in m.get('model_id', '').lower()]
            
            print(f"📋 Total Models: {len(models)}")
            print(f"⚡ Turbo Models: {len(turbo_models)}")
            
            if turbo_models:
                print("✅ Turbo models available (Pro feature)")
                for model in turbo_models:
                    print(f"   • {model.get('name')} ({model.get('model_id')})")
                return True
            else:
                print("⚠️  No Turbo models found")
                return False
        else:
            print(f"❌ Failed to get models: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_pro_audio_quality():
    """Test Pro audio quality features"""
    print("\n🎧 Testing Pro Audio Quality...")
    
    api_key = os.getenv("ELEVENLABS_API_KEY")
    voice_id = "21m00Tcm4TlvDq8ikWAM"  # Rachel voice
    
    headers = {
        "xi-api-key": api_key,
        "Content-Type": "application/json"
    }
    
    # Test different quality settings
    tests = [
        {
            "name": "192 kbps (Pro)",
            "data": {
                "text": "Testing Pro audio quality.",
                "model_id": "eleven_turbo_v2",
                "output_format": "mp3_44100_192"
            }
        },
        {
            "name": "44.1kHz PCM (Pro)",
            "data": {
                "text": "Testing PCM audio output.",
                "model_id": "eleven_turbo_v2", 
                "output_format": "pcm_44100"
            }
        }
    ]
    
    for test in tests:
        print(f"   Testing {test['name']}...")
        
        try:
            url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
            response = requests.post(url, headers=headers, json=test['data'])
            
            if response.status_code == 200:
                audio_size = len(response.content)
                print(f"      ✅ Success - {audio_size:,} bytes")
                
                # Save test file
                filename = f"pro_test_{test['name'].replace(' ', '_').replace('(', '').replace(')', '').replace('.', '_')}.mp3"
                with open(filename, 'wb') as f:
                    f.write(response.content)
                print(f"      💾 Saved as: {filename}")
            else:
                print(f"      ❌ Failed: {response.status_code}")
                
        except Exception as e:
            print(f"      ❌ Error: {e}")

async def test_concurrent_requests():
    """Test Pro concurrent request capability"""
    print("\n🔄 Testing Concurrent Requests...")
    
    api_key = os.getenv("ELEVENLABS_API_KEY")
    voice_id = "21m00Tcm4TlvDq8ikWAM"
    
    headers = {
        "xi-api-key": api_key,
        "Content-Type": "application/json"
    }
    
    async def make_request(request_id):
        """Make a single TTS request"""
        start_time = time.time()
        
        try:
            url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
            data = {
                "text": f"Concurrent request {request_id}",
                "model_id": "eleven_turbo_v2"
            }
            
            response = requests.post(url, headers=headers, json=data)
            duration = time.time() - start_time
            
            if response.status_code == 200:
                print(f"   ✅ Request {request_id}: {duration:.2f}s")
                return True, duration
            else:
                print(f"   ❌ Request {request_id}: Failed ({response.status_code})")
                return False, 0
                
        except Exception as e:
            print(f"   ❌ Request {request_id}: Error ({e})")
            return False, 0
    
    # Test 10 concurrent requests (Pro limit)
    print("   Testing 10 concurrent requests...")
    tasks = [make_request(i) for i in range(1, 11)]
    
    start_time = time.time()
    results = await asyncio.gather(*tasks)
    total_time = time.time() - start_time
    
    successful = sum(1 for success, _ in results if success)
    avg_duration = sum(duration for _, duration in results if duration > 0) / max(successful, 1)
    
    print(f"\n📊 Results:")
    print(f"   Successful: {successful}/10")
    print(f"   Total Time: {total_time:.2f}s")
    print(f"   Average Request: {avg_duration:.2f}s")
    
    if successful >= 8:
        print("   ✅ Pro concurrent processing working")
    else:
        print("   ⚠️  Some requests failed")

def main():
    """Run all Pro tests"""
    print("🚀 ElevenLabs Pro Features Test")
    print("=" * 40)
    
    # Test 1: Subscription
    if not test_pro_subscription():
        print("\n❌ Cannot proceed without Pro subscription")
        return
    
    # Test 2: Models
    test_pro_models()
    
    # Test 3: Audio Quality
    test_pro_audio_quality()
    
    # Test 4: Concurrent Requests
    asyncio.run(test_concurrent_requests())
    
    print("\n✅ Pro features test completed!")
    print("🎯 You can now use pro_voice_changer.py for optimal performance")

if __name__ == "__main__":
    main() 