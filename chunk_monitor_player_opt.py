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
import subprocess
import sys
from datetime import datetime

def timestamp():
    """Get current timestamp in HH:MM:SS.mmm format"""
    return datetime.now().strftime("%H:%M:%S.%f")[:-3]

def log(message):
    """Print message with timestamp and save to log file"""
    timestamp_str = timestamp()
    log_message = f"[{timestamp_str}] {message}"
    
    # Print to console
    print(log_message)
    
    # Save to log file
    try:
        log_filename = f"chunk_monitor_player_{time.strftime('%Y%m%d_%H%M%S')}.log"
        with open(log_filename, 'a', encoding='utf-8') as log_file:
            log_file.write(log_message + '\n')
            log_file.flush()  # Ensure immediate write
    except Exception as e:
        # Don't use log() here to avoid infinite recursion
        print(f"[{timestamp_str}] ❌ Error writing to log file: {e}")

# Try to import pygame and pydub
PLAYSOUND_AVAILABLE = False
PYDUB_AVAILABLE = False
PYGAME_AVAILABLE = False

try:
    import pygame
    # Optimize pygame mixer for low latency
    pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=256)  # Reduced buffer from 512 to 256
    pygame.mixer.set_reserved(1)  # Reserve a channel for immediate playback
    PYGAME_AVAILABLE = True
    log("✅ pygame library loaded successfully (optimized for low latency)")
except ImportError as e:
    log("⚠️ pygame not found. Attempting to install...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pygame"])
        import pygame
        pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=256)
        pygame.mixer.set_reserved(1)
        PYGAME_AVAILABLE = True
        log("✅ pygame installed and loaded successfully (optimized for low latency)")
    except Exception as install_error:
        log(f"❌ Failed to install pygame: {install_error}")
        log("Please install manually with: pip install pygame")

try:
    from pydub import AudioSegment
    PYDUB_AVAILABLE = True
    log("✅ pydub library loaded successfully")
except ImportError as e:
    log("⚠️ pydub not found. Attempting to install...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pydub"])
        from pydub import AudioSegment
        PYDUB_AVAILABLE = True
        log("✅ pydub installed and loaded successfully")
    except Exception as install_error:
        log(f"❌ Failed to install pydub: {install_error}")
        log("Please install manually with: pip install pydub")

if PYGAME_AVAILABLE and PYDUB_AVAILABLE:
    log("✅ All required libraries loaded successfully")
else:
    log("⚠️ Some libraries are missing. Audio playback may not work properly.")

# Configure logging with timestamps
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S.%f',
    handlers=[
        logging.FileHandler('chunk_player.log'),
        logging.StreamHandler()
    ]
)

class ChunkMonitorPlayer:
    def __init__(self, folder_path="sts_result_chunks", check_interval=0.2, volume_boost=10):  # Reduced from 1.0s to 0.2s
        self.folder_path = Path(folder_path)
        self.check_interval = check_interval
        self.volume_boost = volume_boost  # Volume boost in dB
        self.played_files = set()
        self.audio_queue = deque()
        self.is_playing = False
        self.should_stop = False
        
        # Create folder if it doesn't exist
        self.folder_path.mkdir(exist_ok=True)
        
        log(f"🎵 Chunk Monitor Player initialized for folder: {self.folder_path}")
        log(f"🔊 Volume boost: {self.volume_boost} dB")
        log(f"⏱️  Check interval: {self.check_interval}s (optimized for low latency)")
    
    def get_chunk_number(self, filename):
        """Extract chunk number from filename like 'sts_chunk_001_20250714_101458.mp3'"""
        match = re.search(r'sts_chunk_(\d+)_', filename)
        if match:
            return int(match.group(1))
        return 0
    
    def get_sorted_chunks(self):
        """Get all WAV files in the folder sorted by chunk number"""
        pattern = str(self.folder_path / "sts_chunk_*.wav")  # Changed from .mp3 to .wav
        files = glob.glob(pattern)
        
        # Sort by chunk number
        sorted_files = sorted(files, key=lambda x: self.get_chunk_number(os.path.basename(x)))
        return sorted_files
    
    def play_audio_file(self, file_path):
        """Play a single audio file with volume boost (optimized for low latency)"""
        try:
            filename = os.path.basename(file_path)
            start_time = time.time()
            
            # Quick check if file exists and has content
            if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
                log(f"⚠️ File not ready: {filename}")
                return
            
            if PYGAME_AVAILABLE:
                # Option 1: Direct WAV playback (fastest - no temp files)
                if self.volume_boost == 0:  # No volume boost needed
                    log(f"▶️  Direct WAV playback: {filename}")
                    # Use absolute path to avoid pygame path issues
                    abs_file_path = os.path.abspath(file_path)
                    pygame.mixer.music.load(abs_file_path)
                    pygame.mixer.music.play()
                    while pygame.mixer.music.get_busy():
                        time.sleep(0.01)
                    playback_time = time.time() - start_time
                    log(f"✅ Finished playing: {filename} (took {playback_time:.1f}s)")
                    return
                
                # Option 2: Volume boost with temp file (slower but needed for boost)
                if PYDUB_AVAILABLE:
                    log(f"📁 Loading audio file: {filename}")
                    audio = AudioSegment.from_mp3(file_path)
                    
                    # Boost volume
                    log(f"🔊 Boosting volume by {self.volume_boost} dB")
                    boosted_audio = audio + self.volume_boost
                    
                    # Create temporary WAV file (faster than MP3 for pygame)
                    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                        temp_path = temp_file.name
                    
                    log(f"💾 Creating temporary WAV file")
                    boosted_audio.export(temp_path, format='wav', parameters=["-ar", "44100"])
                    
                    log(f"▶️  Starting playback of {filename} (duration: {len(boosted_audio)/1000:.1f}s)")
                    pygame.mixer.music.load(temp_path)
                    pygame.mixer.music.play()
                    
                    while pygame.mixer.music.get_busy():
                        time.sleep(0.01)
                    
                    playback_time = time.time() - start_time
                    log(f"✅ Finished playing: {filename} (took {playback_time:.1f}s)")
                
                    # Clean up temp file
                    try:
                        os.unlink(temp_path)
                        log(f"🧹 Cleaned up temporary file")
                    except Exception as cleanup_error:
                        log(f"⚠️ Failed to clean up temporary file: {cleanup_error}")
                else:
                    log(f"⚠️ pydub not available, playing at original volume: {filename}")
                    abs_file_path = os.path.abspath(file_path)
                    pygame.mixer.music.load(abs_file_path)
                    pygame.mixer.music.play()
                    while pygame.mixer.music.get_busy():
                        time.sleep(0.01)
                    playback_time = time.time() - start_time
                    log(f"✅ Finished playing: {filename} (took {playback_time:.1f}s)")
            else:
                log(f"❌ pygame not available. Cannot play: {filename}")
                log("Please install with: pip install pygame")
                return
                
        except Exception as e:
            log(f"❌ Error playing {file_path}: {e}")
    
    def audio_player_thread(self):
        """Thread function for playing audio from queue (optimized for low latency)"""
        log("🎵 Audio player thread started (optimized)")
        while not self.should_stop:
            if self.audio_queue and not self.is_playing:
                self.is_playing = True
                file_path = self.audio_queue.popleft()
                log(f"🎵 Starting to play: {os.path.basename(file_path)}")
                self.play_audio_file(file_path)
                self.is_playing = False
                log(f"🎵 Finished playing: {os.path.basename(file_path)}")
            else:
                time.sleep(0.01)  # Reduced from 0.05 to 0.01 for faster response
        log("🎵 Audio player thread stopped")
    
    def monitor_and_play(self):
        """Main monitoring function (optimized for low latency)"""
        log("🔍 Starting chunk monitoring and playback (optimized for low latency)...")
        
        # Start audio player thread
        player_thread = threading.Thread(target=self.audio_player_thread, daemon=True)
        player_thread.start()
        log("✅ Audio player thread started")
        
        last_queue_report = 0
        report_interval = 1.0  # Report queue status every 1 second instead of 2 seconds
        file_sizes = {}  # Track file sizes to detect when writing is complete
        
        try:
            while not self.should_stop:
                # Get current files in the folder
                current_files = self.get_sorted_chunks()
                
                # Find new files that haven't been played
                new_files_count = 0
                for file_path in current_files:
                    if file_path not in self.played_files:
                        # Check if file is fully written by monitoring size
                        try:
                            current_size = os.path.getsize(file_path)
                            if file_path in file_sizes:
                                if file_sizes[file_path] == current_size:  # Size stable = file complete
                                    self.played_files.add(file_path)
                                    self.audio_queue.append(file_path)
                                    new_files_count += 1
                                    log(f"📥 Added to queue: {os.path.basename(file_path)}")
                                    del file_sizes[file_path]  # Clean up
                                else:
                                    # Size changed, still being written
                                    file_sizes[file_path] = current_size
                            else:
                                # First time seeing this file
                                file_sizes[file_path] = current_size
                        except OSError:
                            # File might be locked, skip for now
                            continue
                
                if new_files_count > 0:
                    log(f"📊 Found {new_files_count} new files")
                
                # Log queue status less frequently to avoid spam
                current_time = time.time()
                if self.audio_queue and (current_time - last_queue_report) >= report_interval:
                    log(f"📋 Queue length: {len(self.audio_queue)}, Currently playing: {self.is_playing}")
                    last_queue_report = current_time
                
                time.sleep(self.check_interval)
                
        except KeyboardInterrupt:
            log("⏹️  Received interrupt signal, stopping...")
            self.should_stop = True
            
        except Exception as e:
            log(f"❌ Error in monitoring: {e}")
            self.should_stop = True
        
        finally:
            self.should_stop = True
            log("🛑 Chunk Monitor Player stopped.")
    
    def stop(self):
        """Stop the monitor and player"""
        log("🛑 Stopping Chunk Monitor Player...")
        self.should_stop = True

def main():
    """Main function"""
    log("🎵 Chunk Monitor Player")
    log("=" * 50)
    log("This script monitors the 'sts_result_chunks' folder and plays audio chunks in order.")
    log("Press Ctrl+C to stop.")
    log("")
    
    if not PYGAME_AVAILABLE:
        log("❌ ERROR: pygame library not found!")
        log("Please install it with: pip install pygame")
        log("Note: Use version 1.2.2 for better compatibility")
        return
    
    if not PYDUB_AVAILABLE:
        log("WARNING: pydub not available. Audio will play at original volume.")
        log("Install with: pip install pydub")
    
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description="Chunk Monitor Player")
    parser.add_argument("--no-boost", action="store_true", help="Disable volume boost for maximum speed")
    parser.add_argument("--volume", type=int, default=10, help="Volume boost in dB (default: 10)")
    args = parser.parse_args()
    
    # Set volume boost
    volume_boost = 0 if args.no_boost else args.volume
    
    if args.no_boost:
        log("🚀 Speed mode: Volume boost disabled for maximum speed")
    else:
        log(f"🔊 Volume boost: {volume_boost} dB")
    
    # Create and start the monitor
    log(f"🎵 Creating Chunk Monitor Player with {volume_boost} dB volume boost")
    monitor = ChunkMonitorPlayer(volume_boost=volume_boost)
    
    try:
        monitor.monitor_and_play()
    except KeyboardInterrupt:
        log("\n⏹️  Stopping...")
        monitor.stop()

if __name__ == "__main__":
    main() 