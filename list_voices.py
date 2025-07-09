#!/usr/bin/env python3
"""
List all available voices from ElevenLabs
Helps you find a valid Voice ID for the voice changer
"""

import os
from dotenv import load_dotenv
import elevenlabs

def main():
    """List all available voices"""
    print("🎭 ElevenLabs Voice Lister")
    print("=" * 50)
    
    # Load environment variables
    load_dotenv()
    api_key = os.getenv("ELEVENLABS_API_KEY")
    
    if not api_key or api_key == "your_api_key_here":
        print("❌ Please set your ELEVENLABS_API_KEY in the .env file")
        return
    
    try:
        # Set API key
        elevenlabs.set_api_key(api_key)
        
        # Get all voices
        print("🔍 Fetching voices from ElevenLabs...")
        voices = elevenlabs.voices()
        
        print(f"\n✅ Found {len(voices)} voices:")
        print("=" * 50)
        
        for i, voice in enumerate(voices, 1):
            print(f"\n{i}. Voice: {voice.name}")
            print(f"   ID: {voice.voice_id}")
            print(f"   Description: {voice.description}")
            print(f"   Category: {voice.category}")
            print(f"   Labels: {voice.labels}")
            print("-" * 30)
        
        print(f"\n📝 To use a voice in your .env file:")
        print(f"   VOICE_ID=voice_id_from_above")
        
        # Show some recommendations
        print(f"\n💡 Recommendations:")
        print(f"   - Use voices you own for best results")
        print(f"   - Professional clones have highest quality")
        print(f"   - Community voices are free to use")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print(f"   Make sure your API key is correct and has the right permissions")

if __name__ == "__main__":
    main() 