#!/usr/bin/env python3
"""
List available ElevenLabs voices and find Russian ones
"""

import os
from dotenv import load_dotenv
from elevenlabs import voices, set_api_key

# Load environment variables
load_dotenv()

# Setup API key
api_key = os.getenv('ELEVENLABS_API_KEY')
if not api_key:
    print("❌ ELEVENLABS_API_KEY not found in environment variables")
    exit(1)
set_api_key(api_key)

def list_voices():
    """List all available voices"""
    try:
        all_voices = voices()
        print(f"Found {len(all_voices)} voices:")
        print("-" * 80)
        
        russian_voices = []
        
        for voice in all_voices:
            voice_id = voice.voice_id
            name = voice.name
            labels = voice.labels or {}
            language = labels.get('language', 'Unknown')
            
            print(f"ID: {voice_id}")
            print(f"Name: {name}")
            print(f"Language: {language}")
            print(f"Labels: {labels}")
            print("-" * 40)
            
            # Check if it's Russian
            if 'russian' in language.lower() or 'ru' in language.lower():
                russian_voices.append(voice)
        
        print(f"\n🎯 Found {len(russian_voices)} Russian voices:")
        for voice in russian_voices:
            print(f"  - {voice.name} (ID: {voice.voice_id})")
            
    except Exception as e:
        print(f"Error listing voices: {e}")

if __name__ == "__main__":
    list_voices() 