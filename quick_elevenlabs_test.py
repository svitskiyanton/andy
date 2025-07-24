#!/usr/bin/env python3
"""
Quick ElevenLabs connectivity test
"""

import asyncio
import time
import json
import websockets
import os
from dotenv import load_dotenv

load_dotenv()

async def quick_test():
    """Quick test of ElevenLabs API"""
    print("🔍 Quick ElevenLabs Test...")
    
    EL_API_KEY = os.getenv("ELEVENLABS_API_KEY")
    EL_VOICE_ID = os.getenv("EL_VOICE_ID", "21m00Tcm4TlvDq8ikWAM")
    
    try:
        url = f"wss://api.elevenlabs.io/v1/text-to-speech/{EL_VOICE_ID}/multi-stream-input?model_id=eleven_flash_v2_5&output_format=pcm_16000&auto_mode=true"
        
        start_time = time.time()
        
        async with websockets.connect(
            url,
            additional_headers={"xi-api-key": EL_API_KEY}
        ) as websocket:
            connect_time = time.time() - start_time
            print(f"✅ Connected in {connect_time:.3f}s")
            
            # Simple test
            test_message = {
                "text": "Hello test.",
                "voice_settings": {"stability": 0.3, "similarity_boost": 0.7, "speed": 1.1}
            }
            
            tts_start = time.time()
            await websocket.send(json.dumps(test_message))
            print("📤 Text sent")
            
            # Wait for response
            try:
                data = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                response_time = time.time() - tts_start
                print(f"✅ Response in {response_time:.3f}s")
                return True
            except asyncio.TimeoutError:
                print("❌ No response within 5s")
                return False
                
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

async def main():
    """Run multiple quick tests"""
    print("🚀 ElevenLabs Quick Health Check")
    print("=" * 40)
    
    for i in range(3):
        print(f"\n🔄 Test {i+1}/3:")
        success = await quick_test()
        if success:
            print("✅ ElevenLabs is responding normally!")
            break
        else:
            print("❌ Still having issues...")
            if i < 2:  # Don't sleep after last test
                await asyncio.sleep(2.0)
    
    print("\n" + "=" * 40)
    if success:
        print("🎉 ElevenLabs appears to be working now!")
        print("💡 Try running your main script again.")
    else:
        print("⚠️  ElevenLabs still has issues.")
        print("💡 Wait 15-30 minutes and try again.")

if __name__ == "__main__":
    asyncio.run(main()) 