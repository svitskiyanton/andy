#!/usr/bin/env python3
"""
Test transcription functionality
"""

import asyncio
import logging
import os
import tempfile
import requests
import soundfile as sf
import numpy as np
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def test_transcription():
    """Test transcription with a simple audio file"""
    try:
        # Create a simple test audio (sine wave)
        sample_rate = 16000
        duration = 2.0  # 2 seconds
        frequency = 440  # A4 note
        
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        audio = 0.3 * np.sin(2 * np.pi * frequency * t)
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            sf.write(temp_file, audio, sample_rate)
            temp_file_path = temp_file.name
        
        logger.info(f"Created test audio file: {temp_file_path}")
        
        # Test transcription
        headers = {
            "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}"
        }
        
        with open(temp_file_path, "rb") as audio_file:
            files = {
                "file": ("audio.wav", audio_file, "audio/wav"),
                "model": (None, "whisper-1"),
                "language": (None, "ru"),
                "response_format": (None, "json"),
                "temperature": (None, "0.0"),
            }
            
            logger.info("Sending audio to Whisper API...")
            response = requests.post(
                "https://api.openai.com/v1/audio/transcriptions",
                headers=headers,
                files=files
            )
        
        os.unlink(temp_file_path)
        
        logger.info(f"Whisper API response: status={response.status_code}")
        logger.info(f"Response text: {response.text}")
        
        if response.status_code == 200:
            result = response.json()
            text = result.get("text", "").strip()
            logger.info(f"Transcribed text: '{text}'")
            return text
        else:
            logger.error(f"Whisper API error: {response.status_code} - {response.text}")
            return None
            
    except Exception as e:
        logger.error(f"Transcription test error: {e}")
        return None

async def main():
    """Main function"""
    logger.info("Testing transcription...")
    result = await test_transcription()
    if result:
        logger.info("Transcription test successful!")
    else:
        logger.error("Transcription test failed!")

if __name__ == "__main__":
    asyncio.run(main()) 