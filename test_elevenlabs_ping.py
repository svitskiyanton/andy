#!/usr/bin/env python3
"""
Test ElevenLabs API connectivity and response times
"""

import asyncio
import time
import json
import websockets
import base64
import uuid
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

EL_API_KEY = os.getenv("ELEVENLABS_API_KEY")
EL_VOICE_ID = os.getenv("EL_VOICE_ID", "21m00Tcm4TlvDq8ikWAM")

async def test_elevenlabs_connectivity():
    """Test basic ElevenLabs connectivity"""
    print("🔍 Testing ElevenLabs API connectivity...")
    
    try:
        # Test 1: Simple text-to-speech request
        url = f"wss://api.elevenlabs.io/v1/text-to-speech/{EL_VOICE_ID}/multi-stream-input"
        params = {
            "model_id": "eleven_flash_v2_5",
            "output_format": "pcm_16000",
            "auto_mode": "true",
            "inactivity_timeout": "60",
            "sync_alignment": "false"
        }
        
        query_string = "&".join([f"{k}={v}" for k, v in params.items()])
        full_url = f"{url}?{query_string}"
        
        print(f"🔗 Connecting to: {full_url}")
        
        start_time = time.time()
        
        async with websockets.connect(
            full_url,
            additional_headers={"xi-api-key": EL_API_KEY}
        ) as websocket:
            connect_time = time.time() - start_time
            print(f"✅ Connected in {connect_time:.3f}s")
            
            # Initialize context
            context_id = f"test_{uuid.uuid4().hex[:8]}"
            init_message = {
                "text": " ",
                "voice_settings": {
                    "stability": 0.3,
                    "similarity_boost": 0.7,
                    "speed": 1.1
                },
                "context_id": context_id
            }
            
            await websocket.send(json.dumps(init_message))
            print("✅ Context initialized")
            
            # Test simple text
            test_text = "Hello, this is a test."
            text_message = {"text": test_text, "xi_api_key": EL_API_KEY}
            
            tts_start = time.time()
            await websocket.send(json.dumps(text_message))
            print(f"📤 Text sent: '{test_text}'")
            
            # Wait for audio response
            audio_received = False
            timeout = 10.0
            start_wait = time.time()
            
            while (time.time() - start_wait) < timeout:
                try:
                    data = await asyncio.wait_for(websocket.recv(), timeout=0.1)
                    data = json.loads(data)
                    
                    if "audio" in data:
                        audio_received = True
                        response_time = time.time() - tts_start
                        print(f"✅ Audio received in {response_time:.3f}s")
                        print(f"📊 Audio length: {len(data['audio'])} chars")
                        break
                        
                except asyncio.TimeoutError:
                    continue
                except Exception as e:
                    print(f"❌ Error receiving data: {e}")
                    break
            
            if not audio_received:
                print(f"❌ No audio received within {timeout}s")
            
            # Test multiple requests
            print("\n🔄 Testing multiple requests...")
            for i in range(3):
                test_text = f"Test message number {i+1}."
                text_message = {"text": test_text, "xi_api_key": EL_API_KEY}
                
                tts_start = time.time()
                await websocket.send(json.dumps(text_message))
                
                audio_received = False
                start_wait = time.time()
                
                while (time.time() - start_wait) < 5.0:  # 5 second timeout per request
                    try:
                        data = await asyncio.wait_for(websocket.recv(), timeout=0.1)
                        data = json.loads(data)
                        
                        if "audio" in data:
                            response_time = time.time() - tts_start
                            print(f"✅ Request {i+1}: {response_time:.3f}s")
                            break
                            
                    except asyncio.TimeoutError:
                        continue
                    except Exception as e:
                        print(f"❌ Request {i+1} error: {e}")
                        break
                
                if not audio_received:
                    print(f"❌ Request {i+1}: No response within 5s")
                
                await asyncio.sleep(1.0)  # Wait between requests
            
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return False
    
    return True

async def test_network_latency():
    """Test basic network latency to ElevenLabs"""
    print("\n🌐 Testing network latency...")
    
    import socket
    
    try:
        # Test DNS resolution
        start_time = time.time()
        ip = socket.gethostbyname("api.elevenlabs.io")
        dns_time = time.time() - start_time
        print(f"✅ DNS resolution: {dns_time:.3f}s ({ip})")
        
        # Test TCP connection
        start_time = time.time()
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5.0)
        result = sock.connect_ex(('api.elevenlabs.io', 443))
        tcp_time = time.time() - start_time
        sock.close()
        
        if result == 0:
            print(f"✅ TCP connection: {tcp_time:.3f}s")
        else:
            print(f"❌ TCP connection failed: {result}")
            
    except Exception as e:
        print(f"❌ Network test failed: {e}")

async def main():
    """Main test function"""
    print("🚀 ElevenLabs API Health Check")
    print(f"🔑 API Key: {'*' * (len(EL_API_KEY) - 4) + EL_API_KEY[-4:] if EL_API_KEY else 'NOT FOUND'}")
    print(f"🎤 Voice ID: {EL_VOICE_ID}")
    print("=" * 50)
    
    # Test network first
    await test_network_latency()
    
    # Test ElevenLabs API
    success = await test_elevenlabs_connectivity()
    
    print("\n" + "=" * 50)
    if success:
        print("✅ ElevenLabs API appears to be working")
    else:
        print("❌ ElevenLabs API has issues")
    
    print("\n💡 If you see delays > 1 second, it might be:")
    print("   - ElevenLabs server issues")
    print("   - Network congestion")
    print("   - API rate limiting")
    print("   - Account/billing issues")

if __name__ == "__main__":
    asyncio.run(main()) 