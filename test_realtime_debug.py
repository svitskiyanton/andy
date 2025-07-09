#!/usr/bin/env python3
"""
Debug Test for Real-time Voice Changer
Simple test to see what's happening with the components
"""

import os
import threading
import time
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class DebugVoiceChanger:
    def __init__(self):
        self.TEXT_FILE = "voice_changer_text.txt"
        self.running = True
        self.last_file_size = 0
        self.last_line_count = 0
    
    def file_monitor_worker(self):
        """Monitor text file for new content"""
        print("📁 File Monitor: Starting...")
        
        while self.running:
            try:
                if os.path.exists(self.TEXT_FILE):
                    current_size = os.path.getsize(self.TEXT_FILE)
                    
                    if current_size > self.last_file_size:
                        print(f"📁 File Monitor: New content! Size: {self.last_file_size} → {current_size}")
                        
                        with open(self.TEXT_FILE, 'r', encoding='utf-8') as f:
                            lines = f.readlines()
                        
                        print(f"📁 File Monitor: Found {len(lines)} lines, processing from {self.last_line_count}")
                        
                        new_lines = lines[self.last_line_count:]
                        for line in new_lines:
                            if line.strip():
                                print(f"🎵 TTS: Would process: '{line.strip()}'")
                        
                        self.last_line_count = len(lines)
                        self.last_file_size = current_size
                    else:
                        print(f"📁 File Monitor: Monitoring... (size: {current_size})")
                else:
                    print("📁 File Monitor: Waiting for file...")
                
                time.sleep(1)  # Check every second
                
            except Exception as e:
                print(f"❌ File monitor error: {e}")
                time.sleep(1)
    
    def test_writer(self):
        """Test writing to file"""
        print("📝 Test Writer: Starting...")
        
        for i in range(5):
            if not self.running:
                break
                
            test_text = f"Test message {i+1}: Hello world!"
            print(f"📝 Writing: '{test_text}'")
            
            with open(self.TEXT_FILE, 'a', encoding='utf-8') as f:
                f.write(test_text + "\n")
            
            time.sleep(3)  # Wait 3 seconds between writes
    
    def start(self):
        """Start debug test"""
        print("🔍 Debug Test: Real-time Voice Changer")
        print("=" * 40)
        
        # Clear text file
        if os.path.exists(self.TEXT_FILE):
            os.remove(self.TEXT_FILE)
        
        # Start file monitor thread
        monitor_thread = threading.Thread(target=self.file_monitor_worker)
        monitor_thread.daemon = True
        monitor_thread.start()
        
        print("✅ File monitor started")
        
        # Start test writer
        writer_thread = threading.Thread(target=self.test_writer)
        writer_thread.daemon = True
        writer_thread.start()
        
        print("✅ Test writer started")
        print("⏹️  Press Ctrl+C to stop")
        
        try:
            while self.running:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n⏹️  Stopping debug test...")
            self.running = False

def main():
    """Main function"""
    print("🔍 Starting debug test...")
    
    debug_test = DebugVoiceChanger()
    debug_test.start()

if __name__ == "__main__":
    main() 