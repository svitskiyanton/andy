#!/usr/bin/env python3
"""
Chunk Monitor Player
Monitors the sts_result_chunks folder and plays audio chunks in sequential order.
"""

import os
import time
import glob
import re
from pathlib import Path
import threading
from collections import deque
import logging
import tempfile

# Try to import playsound and pydub
try:
    from playsound import playsound
    from pydub import AudioSegment
    PLAYSOUND_AVAILABLE = True
    PYDUB_AVAILABLE = True
except ImportError as e:
    if "playsound" in str(e):
        PLAYSOUND_AVAILABLE = False
        print("playsound not found. Installing...")
        print("Run: pip install playsound==1.2.2")
    if "pydub" in str(e):
        PYDUB_AVAILABLE = False
        print("pydub not found. Installing...")
        print("Run: pip install pydub")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('chunk_player.log'),
        logging.StreamHandler()
    ]
)

class ChunkMonitorPlayer:
    def __init__(self, folder_path="sts_result_chunks", check_interval=1.0, volume_boost=10):
        self.folder_path = Path(folder_path)
        self.check_interval = check_interval
        self.volume_boost = volume_boost  # Volume boost in dB
        self.played_files = set()
        self.audio_queue = deque()
        self.is_playing = False
        self.should_stop = False
        
        # Create folder if it doesn't exist
        self.folder_path.mkdir(exist_ok=True)
        
        logging.info(f"Chunk Monitor Player initialized for folder: {self.folder_path}")
        logging.info(f"Volume boost: {self.volume_boost} dB")
    
    def get_chunk_number(self, filename):
        """Extract chunk number from filename like 'sts_chunk_001_20250714_101458.mp3'"""
        match = re.search(r'sts_chunk_(\d+)_', filename)
        if match:
            return int(match.group(1))
        return 0
    
    def get_sorted_chunks(self):
        """Get all MP3 files in the folder sorted by chunk number"""
        pattern = str(self.folder_path / "sts_chunk_*.mp3")
        files = glob.glob(pattern)
        
        # Sort by chunk number
        sorted_files = sorted(files, key=lambda x: self.get_chunk_number(os.path.basename(x)))
        return sorted_files
    
    def play_audio_file(self, file_path):
        """Play a single audio file with volume boost"""
        try:
            logging.info(f"Playing: {os.path.basename(file_path)}")
            
            if PLAYSOUND_AVAILABLE and PYDUB_AVAILABLE:
                # Load audio with pydub
                audio = AudioSegment.from_mp3(file_path)
                
                # Boost volume
                boosted_audio = audio + self.volume_boost
                
                # Create temporary file for the boosted audio
                with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as temp_file:
                    temp_path = temp_file.name
                
                # Export boosted audio to temporary file
                boosted_audio.export(temp_path, format='mp3')
                
                # Play the boosted audio
                playsound(temp_path, block=True)
                
                # Clean up temporary file
                try:
                    os.unlink(temp_path)
                except:
                    pass
                    
            elif PLAYSOUND_AVAILABLE:
                # Fallback to original playsound without volume boost
                playsound(file_path, block=True)
            else:
                logging.error("playsound not available. Please install with: pip install playsound==1.2.2")
                return
                
        except Exception as e:
            logging.error(f"Error playing {file_path}: {e}")
    
    def audio_player_thread(self):
        """Thread function for playing audio from queue"""
        while not self.should_stop:
            if self.audio_queue and not self.is_playing:
                self.is_playing = True
                file_path = self.audio_queue.popleft()
                self.play_audio_file(file_path)
                self.is_playing = False
            else:
                time.sleep(0.1)
    
    def monitor_and_play(self):
        """Main monitoring function"""
        logging.info("Starting chunk monitoring and playback...")
        
        # Start audio player thread
        player_thread = threading.Thread(target=self.audio_player_thread, daemon=True)
        player_thread.start()
        
        try:
            while not self.should_stop:
                # Get current files in folder
                current_files = self.get_sorted_chunks()
                
                # Find new files that haven't been played
                for file_path in current_files:
                    if file_path not in self.played_files:
                        self.played_files.add(file_path)
                        self.audio_queue.append(file_path)
                        logging.info(f"Added to queue: {os.path.basename(file_path)}")
                
                # Log queue status
                if self.audio_queue:
                    logging.info(f"Queue length: {len(self.audio_queue)}, Currently playing: {self.is_playing}")
                
                time.sleep(self.check_interval)
                
        except KeyboardInterrupt:
            logging.info("Received interrupt signal, stopping...")
            self.should_stop = True
            
        except Exception as e:
            logging.error(f"Error in monitoring: {e}")
            self.should_stop = True
        
        finally:
            self.should_stop = True
            logging.info("Chunk Monitor Player stopped.")
    
    def stop(self):
        """Stop the monitor and player"""
        self.should_stop = True

def main():
    """Main function"""
    print("Chunk Monitor Player")
    print("=" * 50)
    print("This script monitors the 'sts_result_chunks' folder and plays audio chunks in order.")
    print("Press Ctrl+C to stop.")
    print()
    
    if not PLAYSOUND_AVAILABLE:
        print("ERROR: playsound library not found!")
        print("Please install it with: pip install playsound==1.2.2")
        print("Note: Use version 1.2.2 for better compatibility")
        return
    
    if not PYDUB_AVAILABLE:
        print("WARNING: pydub not available. Audio will play at original volume.")
        print("Install with: pip install pydub")
    
    # Create and start the monitor with volume boost
    monitor = ChunkMonitorPlayer(volume_boost=10)  # 10 dB volume boost
    
    try:
        monitor.monitor_and_play()
    except KeyboardInterrupt:
        print("\nStopping...")
        monitor.stop()

if __name__ == "__main__":
    main() 