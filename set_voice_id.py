#!/usr/bin/env python3
"""
Set ElevenLabs Voice ID in environment
"""

import os
from dotenv import load_dotenv

# Load existing .env file
load_dotenv()

def set_voice_id():
    """Set the voice ID in environment"""
    print("Available Russian voices from ElevenLabs:")
    print("1. Ekaterina - professional voice (GN4wbsbejSnGSa1AzjH5)")
    print("2. Nina (N8lIVPsFkvOoqev5Csxo)")
    print("3. Arcadias (Obuyk6KKzg9olSLPaCbl)")
    print("4. вика (gG3dMrB9eyKnDheTs2BW)")
    print("5. приятный бас (h6Nllu4xSN3loa3tmHD0)")
    print("6. Anna - Calm and pleasant (rxEz5E7hIAPk7D3bXwf6)")
    print("7. Soft Female Russian voice (ymDCYd8puC7gYjxIamPt)")
    
    voice_id = input("\nEnter the voice ID you want to use (or press Enter for Ekaterina): ").strip()
    
    if not voice_id:
        voice_id = "GN4wbsbejSnGSa1AzjH5"  # Default to Ekaterina
    
    # Update .env file
    env_file = ".env"
    env_content = ""
    
    # Read existing .env file
    if os.path.exists(env_file):
        with open(env_file, 'r', encoding='utf-8') as f:
            env_content = f.read()
    
    # Check if ELEVENLABS_VOICE_ID already exists
    if "ELEVENLABS_VOICE_ID" in env_content:
        # Replace existing line
        lines = env_content.split('\n')
        new_lines = []
        for line in lines:
            if line.startswith("ELEVENLABS_VOICE_ID="):
                new_lines.append(f"ELEVENLABS_VOICE_ID={voice_id}")
            else:
                new_lines.append(line)
        env_content = '\n'.join(new_lines)
    else:
        # Add new line
        if env_content and not env_content.endswith('\n'):
            env_content += '\n'
        env_content += f"ELEVENLABS_VOICE_ID={voice_id}\n"
    
    # Write back to .env file
    with open(env_file, 'w', encoding='utf-8') as f:
        f.write(env_content)
    
    print(f"\n✅ Voice ID set to: {voice_id}")
    print("The voice ID has been saved to your .env file.")
    print("Restart the voice changer to use the new voice.")

if __name__ == "__main__":
    set_voice_id() 