#!/usr/bin/env python3
"""
Setup script for ElevenLabs API key
"""

import os
import sys

def setup_api_key():
    """Setup ElevenLabs API key"""
    print("=== ElevenLabs API Key Setup ===")
    print()
    print("To use the Pro STS voice changer, you need your ElevenLabs API key.")
    print()
    print("1. Go to https://elevenlabs.io/")
    print("2. Sign in to your account")
    print("3. Go to Profile Settings")
    print("4. Copy your API key")
    print()
    
    # Check if API key is already set
    current_key = os.getenv('ELEVENLABS_API_KEY')
    if current_key and current_key != 'your_api_key_here':
        print(f"Current API key: {current_key[:8]}...{current_key[-4:]}")
        print()
        response = input("Do you want to change it? (y/n): ").lower()
        if response != 'y':
            print("API key setup complete!")
            return
    
    print("Enter your ElevenLabs API key:")
    api_key = input("API Key: ").strip()
    
    if not api_key:
        print("No API key provided. Setup cancelled.")
        return
    
    if api_key == 'your_api_key_here':
        print("Please enter your actual API key, not the placeholder.")
        return
    
    # Set environment variable for current session
    os.environ['ELEVENLABS_API_KEY'] = api_key
    
    print()
    print("API key set for current session!")
    print()
    print("To make this permanent, you can:")
    print("1. Set it as a Windows environment variable")
    print("2. Create a .env file in this directory")
    print("3. Set it each time you run the script")
    print()
    print("Testing API key...")
    
    # Test the API key
    try:
        import aiohttp
        import asyncio
        
        async def test_key():
            async with aiohttp.ClientSession() as session:
                headers = {
                    "xi-api-key": api_key,
                    "Content-Type": "application/json"
                }
                
                async with session.get(
                    "https://api.elevenlabs.io/v1/user/subscription",
                    headers=headers
                ) as response:
                    if response.status == 200:
                        sub_info = await response.json()
                        print(f"[OK] API key works! Subscription: {sub_info.get('tier', 'Unknown')}")
                        return True
                    else:
                        print(f"[ERROR] API key test failed: {response.status}")
                        return False
        
        success = asyncio.run(test_key())
        if success:
            print("Setup complete! You can now run: python test_pro_sts.py")
        else:
            print("Please check your API key and try again.")
    
    except Exception as e:
        print(f"Error testing API key: {e}")

if __name__ == "__main__":
    setup_api_key() 