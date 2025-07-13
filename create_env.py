#!/usr/bin/env python3
"""
Create .env file for ElevenLabs API key
"""

import os

def create_env_file():
    """Create .env file with API key"""
    print("=== Creating .env file ===")
    print()
    print("This will create a .env file with your ElevenLabs API key.")
    print()
    print("1. Go to https://elevenlabs.io/")
    print("2. Sign in to your account")
    print("3. Go to Profile Settings")
    print("4. Copy your API key")
    print()
    
    # Check if .env already exists
    if os.path.exists('.env'):
        print("Found existing .env file:")
        with open('.env', 'r') as f:
            content = f.read()
            print(content)
        print()
        response = input("Do you want to overwrite it? (y/n): ").lower()
        if response != 'y':
            print("Keeping existing .env file.")
            return
    
    print("Enter your ElevenLabs API key:")
    api_key = input("API Key: ").strip()
    
    if not api_key:
        print("No API key provided. Setup cancelled.")
        return
    
    if api_key == 'your_api_key_here':
        print("Please enter your actual API key, not the placeholder.")
        return
    
    # Create .env file
    env_content = f"""# ElevenLabs API Configuration
ELEVENLABS_API_KEY={api_key}
VOICE_ID=pNInz6obpgDQGcFmaJgB
# Optional: Set your preferred voice ID above
# Optional: Add GOOGLE_APPLICATION_CREDENTIALS=./google-credentials.json if using Google STT
"""
    
    with open('.env', 'w') as f:
        f.write(env_content)
    
    print()
    print("[OK] .env file created successfully!")
    print()
    print("The .env file contains:")
    print(f"  ELEVENLABS_API_KEY={api_key[:8]}...{api_key[-4:]}")
    print("  VOICE_ID=pNInz6obpgDQGcFmaJgB")
    print()
    print("You can now run:")
    print("  python test_pro_sts.py")
    print("  python pro_speech_to_speech.py")
    print()
    print("Note: The .env file is in .gitignore, so it won't be committed to git.")

if __name__ == "__main__":
    create_env_file() 