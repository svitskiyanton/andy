#!/usr/bin/env python3
"""
Clean up the corrupted text file
"""

import os

def cleanup_file():
    filename = "voice_changer_text.txt"
    
    if not os.path.exists(filename):
        print("File doesn't exist")
        return
    
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            content = f.read()
        
        print(f"Current file content:\n{content}")
        
        # Clear the file
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("")
        
        print("✅ File cleared successfully")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    cleanup_file() 