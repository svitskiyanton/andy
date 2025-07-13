#!/usr/bin/env python3
"""
Simple Audio Level Voice Chunker
================================

A lightweight real-time voice chunker using only audio level detection.
No VAD dependencies required - works with just pyaudio and numpy.

Features:
- Real-time microphone input
- Audio level-based silence detection
- Natural phrase boundary detection
- Configurable parameters
- Minimal dependencies
"""

import os
import sys
import time
import signal
import atexit
import threading
from datetime import datetime
from pathlib import Path

import numpy as np
import pyaudio

class SimpleAudioLevelChunker:
    def __init__(self):
        # Audio settings
        self.SAMPLE_RATE = 44100
        self.CHUNK_SIZE = 2048
        self.CHANNELS = 1
        self.FORMAT = pyaudio.paInt16
        
        # Pause detection settings
        self.SILENCE_THRESHOLD = 0.015  # Increased audio level threshold for silence
        self.SILENCE_DURATION = 1.0  # Increased seconds of silence to end phrase
        self.MIN_PHRASE_DURATION = 1.5  # Increased minimum phrase duration
        self.MAX_PHRASE_DURATION = 10.0  # Maximum phrase duration
        
        # Advanced settings
        self.USE_MOVING_AVERAGE = True  # Use moving average for smoother detection
        self.MOVING_AVERAGE_WINDOW = 10  # Larger window for moving average
        
        # State management
        self.is_running = False
        self.audio = pyaudio.PyAudio()
        self.stream = None
        
        # Audio level history for moving average
        self.audio_level_history = []
        
        # Callback for when phrases are detected
        self.on_phrase_detected = None
        
        print(f"🎤 Simple Audio Level Voice Chunker initialized:")
        print(f"   Sample Rate: {self.SAMPLE_RATE}Hz")
        print(f"   Silence Threshold: {self.SILENCE_THRESHOLD}")
        print(f"   Min Phrase Duration: {self.MIN_PHRASE_DURATION}s")
        print(f"   Max Phrase Duration: {self.MAX_PHRASE_DURATION}s")
        print(f"   Silence Duration: {self.SILENCE_DURATION}s")
        print(f"   Moving Average: {self.USE_MOVING_AVERAGE}")
    
    def set_phrase_callback(self, callback):
        """Set callback function to be called when a phrase is detected"""
        self.on_phrase_detected = callback
    
    def calculate_audio_level(self, audio_data):
        """Calculate audio level with optional moving average smoothing"""
        # Calculate RMS (Root Mean Square) audio level
        audio_level = np.sqrt(np.mean(audio_data.astype(np.float32) ** 2)) / 32768.0
        
        if self.USE_MOVING_AVERAGE:
            # Add to history
            self.audio_level_history.append(audio_level)
            
            # Keep only recent history
            if len(self.audio_level_history) > self.MOVING_AVERAGE_WINDOW:
                self.audio_level_history.pop(0)
            
            # Return moving average
            return np.mean(self.audio_level_history)
        else:
            return audio_level
    
    def start_listening(self):
        """Start listening for speech and chunking phrases"""
        try:
            # Open audio stream
            self.stream = self.audio.open(
                format=self.FORMAT,
                channels=self.CHANNELS,
                rate=self.SAMPLE_RATE,
                input=True,
                frames_per_buffer=self.CHUNK_SIZE
            )
            
            print("🎤 Started listening for speech...")
            print("   Speak naturally - phrases will be detected on silence")
            print("   Press Ctrl+C to stop")
            
            self.is_running = True
            
            # Start the listening thread
            self.listen_thread = threading.Thread(target=self._listen_loop)
            self.listen_thread.daemon = True
            self.listen_thread.start()
            
            # Wait for the thread to complete
            self.listen_thread.join()
            
        except Exception as e:
            print(f"❌ Error starting audio stream: {e}")
        finally:
            self.stop_listening()
    
    def _listen_loop(self):
        """Main listening loop for real-time phrase detection"""
        audio_buffer = b""
        silence_start = None
        phrase_start_time = time.time()
        phrase_count = 0
        last_audio_level = 0
        
        while self.is_running:
            try:
                # Read audio chunk
                data = self.stream.read(self.CHUNK_SIZE, exception_on_overflow=False)
                audio_buffer += data
                
                # Calculate audio level
                audio_data = np.frombuffer(data, dtype=np.int16)
                audio_level = self.calculate_audio_level(audio_data)
                
                current_time = time.time()
                phrase_duration = current_time - phrase_start_time
                
                # Detect silence using audio level
                is_silence = audio_level < self.SILENCE_THRESHOLD
                
                if is_silence:
                    if silence_start is None:
                        silence_start = current_time
                        print(f"🔇 Silence detected (level: {audio_level:.4f})")
                else:
                    if silence_start is not None:
                        print(f"🎤 Speech detected (level: {audio_level:.4f})")
                    silence_start = None
                
                # Check if phrase should end
                should_end = False
                end_reason = ""
                
                # End if silence detected for required duration
                if (len(audio_buffer) > 0 and 
                    phrase_duration >= self.MIN_PHRASE_DURATION and
                    silence_start is not None and
                    (current_time - silence_start) >= self.SILENCE_DURATION):
                    should_end = True
                    end_reason = f"silence detected for {current_time - silence_start:.1f}s"
                
                # End if maximum duration reached
                elif phrase_duration >= self.MAX_PHRASE_DURATION and len(audio_buffer) > 0:
                    should_end = True
                    end_reason = f"max duration reached ({self.MAX_PHRASE_DURATION}s)"
                
                if should_end:
                    phrase_count += 1
                    duration = len(audio_buffer) / (self.SAMPLE_RATE * 2)  # 16-bit = 2 bytes
                    
                    print(f"🎤 Phrase {phrase_count} detected: {duration:.2f}s ({end_reason})")
                    
                    # Call callback if set
                    if self.on_phrase_detected:
                        try:
                            self.on_phrase_detected(audio_buffer, phrase_count, duration)
                        except Exception as e:
                            print(f"❌ Error in phrase callback: {e}")
                    
                    # Reset for next phrase
                    audio_buffer = b""
                    silence_start = None
                    phrase_start_time = current_time
                
                # Small delay to prevent CPU overload
                time.sleep(0.01)
                
            except Exception as e:
                print(f"❌ Error in listening loop: {e}")
                time.sleep(0.1)
    
    def stop_listening(self):
        """Stop listening and cleanup"""
        self.is_running = False
        
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        
        if self.audio:
            self.audio.terminate()
        
        print("🛑 Stopped listening")

def save_phrase_to_file(audio_buffer, phrase_num, duration):
    """Example callback function to save phrases to files"""
    try:
        # Create output directory
        output_dir = Path("audio_level_chunks")
        output_dir.mkdir(exist_ok=True)
        
        # Save as WAV file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = output_dir / f"phrase_{phrase_num:02d}_{timestamp}.wav"
        
        # Convert to numpy array and save
        audio_array = np.frombuffer(audio_buffer, dtype=np.int16)
        
        # Save using soundfile (if available) or wave module
        try:
            import soundfile as sf
            sf.write(str(filename), audio_array, 44100)
        except ImportError:
            # Fallback to wave module
            import wave
            with wave.open(str(filename), 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)  # 16-bit
                wf.setframerate(44100)
                wf.writeframes(audio_buffer)
        
        print(f"💾 Saved phrase {phrase_num}: {filename}")
        
    except Exception as e:
        print(f"❌ Error saving phrase {phrase_num}: {e}")

def print_phrase_info(audio_buffer, phrase_num, duration):
    """Example callback function to just print phrase info"""
    print(f"📊 Phrase {phrase_num}: {len(audio_buffer)} bytes, {duration:.2f}s")

def main():
    """Main function with example usage"""
    print("🎤 Simple Audio Level Voice Chunker")
    print("=" * 50)
    
    # Create chunker
    chunker = SimpleAudioLevelChunker()
    
    # Set callback to save phrases
    chunker.set_phrase_callback(save_phrase_to_file)
    
    # Setup signal handlers for graceful shutdown
    def signal_handler(signum, frame):
        print("\n🛑 Received interrupt signal, stopping...")
        chunker.stop_listening()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    atexit.register(chunker.stop_listening)
    
    try:
        # Start listening
        chunker.start_listening()
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
    finally:
        chunker.stop_listening()
        print("✅ Cleanup completed")

if __name__ == "__main__":
    main() 