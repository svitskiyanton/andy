#!/usr/bin/env python3
"""
Test Pro Speech-to-Speech capabilities
"""

import asyncio
import os
import aiohttp
import json
import logging

# Load environment variables from .env file if it exists
def load_env():
    """Load environment variables from .env file"""
    env_file = '.env'
    if os.path.exists(env_file):
        with open(env_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key.strip()] = value.strip()

# Load .env file
load_env()

from pro_speech_to_speech import ProSTSConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_pro_capabilities():
    """Test Pro subscription and STS capabilities"""
    config = ProSTSConfig()
    
    if not config.ELEVENLABS_API_KEY:
        logger.error("ELEVENLABS_API_KEY environment variable is required")
        return
    
    logger.info("Testing Pro STS capabilities...")
    
    try:
        async with aiohttp.ClientSession() as session:
            headers = {
                "xi-api-key": config.ELEVENLABS_API_KEY,
                "Content-Type": "application/json"
            }
            
            # Test 1: Subscription info
            logger.info("1. Testing subscription...")
            async with session.get(
                "https://api.elevenlabs.io/v1/user/subscription",
                headers=headers
            ) as response:
                if response.status == 200:
                    sub_info = await response.json()
                    logger.info(f"[OK] Subscription: {sub_info.get('tier', 'Unknown')}")
                    logger.info(f"   Character count: {sub_info.get('character_count', 0)}")
                    logger.info(f"   Character limit: {sub_info.get('character_limit', 0)}")
                else:
                    logger.error(f"[ERROR] Subscription test failed: {response.status}")
            
            # Test 2: Available models
            logger.info("2. Testing available models...")
            async with session.get(
                "https://api.elevenlabs.io/v1/models",
                headers=headers
            ) as response:
                if response.status == 200:
                    models = await response.json()
                    turbo_models = [m for m in models if "turbo" in m.get("name", "").lower()]
                    logger.info(f"[OK] Available Turbo models: {len(turbo_models)}")
                    for model in turbo_models:
                        logger.info(f"   - {model['name']} ({model['model_id']})")
                else:
                    logger.error(f"[ERROR] Models test failed: {response.status}")
            
            # Test 3: Available voices
            logger.info("3. Testing available voices...")
            async with session.get(
                "https://api.elevenlabs.io/v1/voices",
                headers=headers
            ) as response:
                if response.status == 200:
                    voices = await response.json()
                    logger.info(f"[OK] Available voices: {len(voices.get('voices', []))}")
                    for voice in voices.get('voices', [])[:5]:  # Show first 5
                        logger.info(f"   - {voice.get('name', 'Unknown')} ({voice.get('voice_id', 'N/A')})")
                else:
                    logger.error(f"[ERROR] Voices test failed: {response.status}")
            
            # Test 4: STS endpoint availability
            logger.info("4. Testing STS endpoint...")
            async with session.get(
                "https://api.elevenlabs.io/v1/speech-to-speech",
                headers=headers
            ) as response:
                if response.status in [200, 405]:  # 405 is expected for GET
                    logger.info("[OK] STS endpoint is available")
                else:
                    logger.error(f"[ERROR] STS endpoint test failed: {response.status}")
            
            # Test 5: Pro features
            logger.info("5. Testing Pro features...")
            logger.info(f"   - Max concurrent requests: {config.MAX_CONCURRENT_REQUESTS}")
            logger.info(f"   - Audio quality: {config.AUDIO_QUALITY} kbps")
            logger.info(f"   - Output format: {config.OUTPUT_FORMAT}")
            logger.info(f"   - Priority processing: {config.PRIORITY_PROCESSING}")
            logger.info(f"   - Latency target: {config.LATENCY_TARGET}s")
            
    except Exception as e:
        logger.error(f"❌ Error testing Pro capabilities: {e}")

async def test_audio_devices():
    """Test audio device availability"""
    logger.info("Testing audio devices...")
    
    try:
        import pyaudio
        
        p = pyaudio.PyAudio()
        
        # List input devices
        logger.info("Input devices:")
        for i in range(p.get_device_count()):
            device_info = p.get_device_info_by_index(i)
            if device_info['maxInputChannels'] > 0:
                logger.info(f"   {i}: {device_info['name']}")
        
        # List output devices
        logger.info("Output devices:")
        for i in range(p.get_device_count()):
            device_info = p.get_device_info_by_index(i)
            if device_info['maxOutputChannels'] > 0:
                logger.info(f"   {i}: {device_info['name']}")
        
        p.terminate()
        logger.info("[OK] Audio device test completed")
        
    except ImportError:
        logger.error("[ERROR] PyAudio not installed")
    except Exception as e:
        logger.error(f"[ERROR] Audio device test failed: {e}")

async def main():
    """Main test function"""
    logger.info("=== Pro STS Capability Test ===")
    
    await test_pro_capabilities()
    await test_audio_devices()
    
    logger.info("=== Test Complete ===")
    logger.info("If all tests passed, you can run: python pro_speech_to_speech.py")

if __name__ == "__main__":
    asyncio.run(main()) 