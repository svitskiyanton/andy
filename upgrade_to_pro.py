#!/usr/bin/env python3
"""
ElevenLabs Pro Subscription Upgrade Script
Tests and validates all Pro features including:
- 10 concurrent requests
- 192 kbps audio quality
- 44.1kHz PCM output
- Turbo models
- Priority processing
"""

import os
import asyncio
import json
import time
import requests
import websockets
import base64
from dotenv import load_dotenv
from typing import List, Dict, Any

# Load environment variables
load_dotenv()

class ProFeatureTester:
    def __init__(self):
        self.api_key = os.getenv("ELEVENLABS_API_KEY")
        self.base_url = "https://api.elevenlabs.io/v1"
        self.test_voice_id = "21m00Tcm4TlvDq8ikWAM"  # Rachel voice
        
        if not self.api_key:
            print("❌ ELEVENLABS_API_KEY not found in environment variables")
            print("   Please add it to your .env file")
            return
        
        print("🔍 ElevenLabs Pro Feature Tester")
        print("=" * 50)
    
    def test_api_key_and_subscription(self):
        """Test API key and get subscription details"""
        print("\n🔑 Testing API Key and Subscription...")
        
        headers = {
            "xi-api-key": self.api_key,
            "Content-Type": "application/json"
        }
        
        try:
            # Get user info
            response = requests.get(f"{self.base_url}/user", headers=headers)
            
            if response.status_code == 200:
                user_data = response.json()
                print(f"✅ API Key valid")
                print(f"   User: {user_data.get('first_name', 'Unknown')} {user_data.get('last_name', '')}")
                print(f"   Subscription: {user_data.get('subscription', {}).get('tier', 'Unknown')}")
                print(f"   Character Count: {user_data.get('subscription', {}).get('character_count', 0):,}")
                print(f"   Character Limit: {user_data.get('subscription', {}).get('character_limit', 0):,}")
                
                # Check if it's Pro
                tier = user_data.get('subscription', {}).get('tier', '').lower()
                if 'pro' in tier:
                    print("🎉 Pro subscription detected!")
                    return True
                else:
                    print("⚠️  This appears to be a non-Pro subscription")
                    return False
            else:
                print(f"❌ API Key test failed: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ Error testing API key: {e}")
            return False
    
    def test_voice_models(self):
        """Test available voice models including Pro features"""
        print("\n🎵 Testing Voice Models...")
        
        headers = {
            "xi-api-key": self.api_key,
            "Content-Type": "application/json"
        }
        
        try:
            # Get available models
            response = requests.get(f"{self.base_url}/models", headers=headers)
            
            if response.status_code == 200:
                models = response.json()
                
                print("📋 Available Models:")
                for model in models:
                    model_id = model.get('model_id', 'Unknown')
                    name = model.get('name', 'Unknown')
                    can_be_finetuned = model.get('can_be_finetuned', False)
                    can_do_text_to_speech = model.get('can_do_text_to_speech', False)
                    can_do_voice_conversion = model.get('can_do_voice_conversion', False)
                    
                    print(f"   • {name} ({model_id})")
                    print(f"     - TTS: {'✅' if can_do_text_to_speech else '❌'}")
                    print(f"     - Voice Conversion: {'✅' if can_do_voice_conversion else '❌'}")
                    print(f"     - Fine-tuning: {'✅' if can_be_finetuned else '❌'}")
                
                # Check for Turbo models (Pro feature)
                turbo_models = [m for m in models if 'turbo' in m.get('model_id', '').lower()]
                if turbo_models:
                    print(f"\n⚡ Turbo Models Available (Pro Feature): {len(turbo_models)}")
                    for model in turbo_models:
                        print(f"   • {model.get('name')} ({model.get('model_id')})")
                else:
                    print("\n⚠️  No Turbo models found")
                
                return True
            else:
                print(f"❌ Failed to get models: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ Error testing models: {e}")
            return False
    
    async def test_concurrent_requests(self):
        """Test Pro concurrent request capability"""
        print("\n🔄 Testing Concurrent Requests (Pro Feature)...")
        
        # Test text for TTS
        test_text = "This is a test of concurrent processing capabilities."
        
        async def make_tts_request(request_id: int):
            """Make a single TTS request"""
            start_time = time.time()
            
            try:
                # Use REST API for concurrent testing
                url = f"{self.base_url}/text-to-speech/{self.test_voice_id}"
                
                headers = {
                    "xi-api-key": self.api_key,
                    "Content-Type": "application/json"
                }
                
                data = {
                    "text": f"Request {request_id}: {test_text}",
                    "model_id": "eleven_turbo_v2",
                    "voice_settings": {
                        "stability": 0.5,
                        "similarity_boost": 0.75
                    }
                }
                
                response = requests.post(url, headers=headers, json=data)
                
                if response.status_code == 200:
                    duration = time.time() - start_time
                    print(f"   ✅ Request {request_id} completed in {duration:.2f}s")
                    return True, duration
                else:
                    print(f"   ❌ Request {request_id} failed: {response.status_code}")
                    return False, 0
                    
            except Exception as e:
                print(f"   ❌ Request {request_id} error: {e}")
                return False, 0
        
        # Test with 10 concurrent requests (Pro limit)
        print("   Testing 10 concurrent requests...")
        tasks = [make_tts_request(i) for i in range(1, 11)]
        
        start_time = time.time()
        results = await asyncio.gather(*tasks)
        total_time = time.time() - start_time
        
        successful_requests = sum(1 for success, _ in results if success)
        avg_duration = sum(duration for _, duration in results if duration > 0) / max(successful_requests, 1)
        
        print(f"\n📊 Concurrent Request Results:")
        print(f"   Successful: {successful_requests}/10")
        print(f"   Total Time: {total_time:.2f}s")
        print(f"   Average Request Time: {avg_duration:.2f}s")
        
        if successful_requests >= 8:
            print("   ✅ Pro concurrent processing working well")
        else:
            print("   ⚠️  Some concurrent requests failed")
        
        return successful_requests >= 8
    
    def test_audio_quality_options(self):
        """Test Pro audio quality options"""
        print("\n🎧 Testing Audio Quality Options (Pro Feature)...")
        
        headers = {
            "xi-api-key": self.api_key,
            "Content-Type": "application/json"
        }
        
        test_text = "Testing high quality audio output."
        
        # Test different quality settings
        quality_tests = [
            {"name": "128 kbps", "settings": {"optimize_streaming_latency": "4"}},
            {"name": "192 kbps", "settings": {"optimize_streaming_latency": "4", "output_format": "mp3_44100_192"}},
            {"name": "44.1kHz PCM", "settings": {"optimize_streaming_latency": "4", "output_format": "pcm_44100"}}
        ]
        
        for test in quality_tests:
            print(f"\n   Testing {test['name']}...")
            
            try:
                url = f"{self.base_url}/text-to-speech/{self.test_voice_id}"
                
                data = {
                    "text": test_text,
                    "model_id": "eleven_turbo_v2",
                    "voice_settings": {
                        "stability": 0.5,
                        "similarity_boost": 0.75
                    }
                }
                
                # Add quality settings
                data.update(test['settings'])
                
                response = requests.post(url, headers=headers, json=data)
                
                if response.status_code == 200:
                    audio_size = len(response.content)
                    print(f"      ✅ Success - Audio size: {audio_size:,} bytes")
                    
                    # Save test audio
                    filename = f"pro_test_{test['name'].replace(' ', '_').replace('.', '_')}.mp3"
                    with open(filename, 'wb') as f:
                        f.write(response.content)
                    print(f"      💾 Saved as: {filename}")
                else:
                    print(f"      ❌ Failed: {response.status_code}")
                    
            except Exception as e:
                print(f"      ❌ Error: {e}")
    
    async def test_websocket_streaming_pro(self):
        """Test Pro WebSocket streaming with enhanced features"""
        print("\n🌐 Testing Pro WebSocket Streaming...")
        
        try:
            # Pro WebSocket URL with enhanced parameters
            uri = f"wss://api.elevenlabs.io/v1/text-to-speech/{self.test_voice_id}/stream-input"
            params = {
                "model_id": "eleven_turbo_v2",
                "optimize_streaming_latency": "4",
                "output_format": "pcm_44100",
                "audio_quality": "192k"
            }
            
            param_str = "&".join([f"{k}={v}" for k, v in params.items()])
            uri = f"{uri}?{param_str}"
            
            headers = {
                "xi-api-key": self.api_key,
                "Content-Type": "application/json"
            }
            
            print(f"   Connecting to: {uri}")
            
            async with websockets.connect(uri, extra_headers=headers) as websocket:
                print("   ✅ WebSocket connected")
                
                # Send test message with Pro features
                message = {
                    "text": "Testing Pro WebSocket streaming with enhanced features.",
                    "try_trigger_generation": True,
                    "voice_settings": {
                        "stability": 0.5,
                        "similarity_boost": 0.75,
                        "style": 0.0,
                        "use_speaker_boost": True
                    }
                }
                
                await websocket.send(json.dumps(message))
                await websocket.send(json.dumps({"text": ""}))
                
                print("   📤 Message sent, receiving audio...")
                
                # Receive audio chunks
                audio_chunks = []
                start_time = time.time()
                
                async for message in websocket:
                    data = json.loads(message)
                    
                    if "audio" in data:
                        audio_data = base64.b64decode(data["audio"])
                        audio_chunks.append(audio_data)
                        print(f"   🎵 Received audio chunk: {len(audio_data)} bytes")
                    
                    if data.get("isFinal"):
                        break
                
                duration = time.time() - start_time
                total_audio_size = sum(len(chunk) for chunk in audio_chunks)
                
                print(f"\n   📊 WebSocket Results:")
                print(f"      Audio chunks: {len(audio_chunks)}")
                print(f"      Total audio size: {total_audio_size:,} bytes")
                print(f"      Duration: {duration:.2f}s")
                
                # Save combined audio
                if audio_chunks:
                    combined_audio = b''.join(audio_chunks)
                    with open("pro_websocket_test.wav", 'wb') as f:
                        f.write(combined_audio)
                    print(f"      💾 Saved as: pro_websocket_test.wav")
                
                print("   ✅ Pro WebSocket streaming test completed")
                return True
                
        except Exception as e:
            print(f"   ❌ WebSocket test failed: {e}")
            return False
    
    def print_pro_benefits_summary(self):
        """Print summary of Pro benefits"""
        print("\n" + "=" * 60)
        print("🎉 ELEVENLABS PRO SUBSCRIPTION BENEFITS")
        print("=" * 60)
        print("📊 Credits & Usage:")
        print("   • 500,000 credits/month (vs 100k in Creator)")
        print("   • ~500 minutes of high-quality TTS")
        print("   • Additional credits: $0.12/1000")
        print()
        print("⚡ Performance:")
        print("   • 10 concurrent requests (vs 5 in Creator)")
        print("   • Priority processing")
        print("   • Turbo/Flash models for faster generation")
        print()
        print("🎧 Audio Quality:")
        print("   • 192 kbps audio quality (vs 128 kbps)")
        print("   • 44.1kHz PCM audio output via API")
        print("   • Enhanced voice settings")
        print()
        print("🔧 Advanced Features:")
        print("   • Professional Voice Cloning")
        print("   • Usage-based billing for additional credits")
        print("   • Enhanced concurrency limits")
        print("   • Better API response times")
        print("=" * 60)
    
    async def run_all_tests(self):
        """Run all Pro feature tests"""
        print("🚀 Starting ElevenLabs Pro Feature Tests...")
        
        # Test 1: API Key and Subscription
        if not self.test_api_key_and_subscription():
            print("❌ Cannot proceed without valid Pro subscription")
            return
        
        # Test 2: Voice Models
        self.test_voice_models()
        
        # Test 3: Concurrent Requests
        await self.test_concurrent_requests()
        
        # Test 4: Audio Quality Options
        self.test_audio_quality_options()
        
        # Test 5: WebSocket Streaming
        await self.test_websocket_streaming_pro()
        
        # Print summary
        self.print_pro_benefits_summary()
        
        print("\n✅ All Pro feature tests completed!")
        print("🎯 You can now use the pro_voice_changer.py for optimal performance")

def main():
    """Main entry point"""
    tester = ProFeatureTester()
    
    if not tester.api_key:
        return
    
    try:
        asyncio.run(tester.run_all_tests())
    except KeyboardInterrupt:
        print("\n⏹️  Tests interrupted by user")
    except Exception as e:
        print(f"❌ Test error: {e}")

if __name__ == "__main__":
    main() 