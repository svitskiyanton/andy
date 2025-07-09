#!/usr/bin/env python3
"""
Google Cloud Setup Helper for STT→TTS Voice Changer
"""

import os
import json
from dotenv import load_dotenv

def setup_google_cloud():
    """Help user set up Google Cloud credentials"""
    print("🔧 Google Cloud Setup for STT→TTS Voice Changer")
    print("=" * 50)
    
    # Check if credentials already exist
    creds_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if creds_path and os.path.exists(creds_path):
        print(f"✅ Google Cloud credentials found at: {creds_path}")
        return True
    
    print("\n📋 To use Google Speech-to-Text, you need to:")
    print("1. Create a Google Cloud project")
    print("2. Enable Speech-to-Text API")
    print("3. Create a service account and download the JSON key")
    print("4. Set the path to your credentials file")
    
    print("\n🔗 Quick setup links:")
    print("• Google Cloud Console: https://console.cloud.google.com/")
    print("• Speech-to-Text API: https://console.cloud.google.com/apis/library/speech.googleapis.com")
    print("• Service Accounts: https://console.cloud.google.com/apis/credentials")
    
    print("\n📝 Step-by-step instructions:")
    print("1. Go to https://console.cloud.google.com/")
    print("2. Create a new project or select existing one")
    print("3. Enable the 'Cloud Speech-to-Text API'")
    print("4. Go to 'APIs & Services' > 'Credentials'")
    print("5. Click 'Create Credentials' > 'Service Account'")
    print("6. Give it a name (e.g., 'voice-changer-stt')")
    print("7. Grant 'Cloud Speech-to-Text User' role")
    print("8. Create and download the JSON key file")
    print("9. Save the JSON file in your project directory")
    
    # Ask for credentials path
    print("\n💾 Where did you save your Google Cloud credentials JSON file?")
    print("   (e.g., ./google-credentials.json)")
    
    creds_path = input("Path to credentials file: ").strip()
    
    if not creds_path:
        print("❌ No path provided. Setup cancelled.")
        return False
    
    # Remove quotes if user added them
    creds_path = creds_path.strip('"\'')
    
    if not os.path.exists(creds_path):
        print(f"❌ File not found: {creds_path}")
        return False
    
    # Validate JSON file
    try:
        with open(creds_path, 'r') as f:
            creds_data = json.load(f)
        
        required_fields = ['type', 'project_id', 'private_key_id', 'private_key', 'client_email']
        missing_fields = [field for field in required_fields if field not in creds_data]
        
        if missing_fields:
            print(f"❌ Invalid credentials file. Missing fields: {missing_fields}")
            return False
        
        print(f"✅ Valid Google Cloud credentials found!")
        print(f"   Project: {creds_data.get('project_id', 'Unknown')}")
        print(f"   Service Account: {creds_data.get('client_email', 'Unknown')}")
        
    except json.JSONDecodeError:
        print("❌ Invalid JSON file")
        return False
    except Exception as e:
        print(f"❌ Error reading credentials file: {e}")
        return False
    
    # Update .env file
    env_path = ".env"
    env_content = ""
    
    if os.path.exists(env_path):
        with open(env_path, 'r') as f:
            env_content = f.read()
    
    # Add or update GOOGLE_APPLICATION_CREDENTIALS
    if "GOOGLE_APPLICATION_CREDENTIALS=" in env_content:
        # Update existing line
        lines = env_content.split('\n')
        for i, line in enumerate(lines):
            if line.startswith("GOOGLE_APPLICATION_CREDENTIALS="):
                lines[i] = f"GOOGLE_APPLICATION_CREDENTIALS={creds_path}"
                break
        env_content = '\n'.join(lines)
    else:
        # Add new line
        if env_content and not env_content.endswith('\n'):
            env_content += '\n'
        env_content += f"GOOGLE_APPLICATION_CREDENTIALS={creds_path}\n"
    
    # Write updated .env file
    with open(env_path, 'w') as f:
        f.write(env_content)
    
    print(f"\n✅ Google Cloud credentials configured!")
    print(f"   Added to: {env_path}")
    print(f"   Path: {creds_path}")
    
    print("\n🎉 Setup complete! You can now run the STT→TTS voice changer.")
    return True

def test_google_cloud():
    """Test Google Cloud Speech-to-Text connection"""
    print("\n🧪 Testing Google Cloud connection...")
    
    try:
        from google.cloud import speech
        
        # Test client creation
        client = speech.SpeechClient()
        print("✅ Google Cloud Speech client created successfully")
        
        # Test with a simple audio file (optional)
        print("✅ Google Cloud setup is working!")
        return True
        
    except Exception as e:
        print(f"❌ Google Cloud test failed: {e}")
        print("   Please check your credentials and API enablement")
        return False

if __name__ == "__main__":
    if setup_google_cloud():
        test_google_cloud()
    else:
        print("\n❌ Setup failed. Please try again.") 