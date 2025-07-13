#!/usr/bin/env python3
"""
Simple Real-time Voice Chunker
==============================

A proven tool for real-time voice chunking with pause detection.
Uses webrtcvad-wheels and audio level detection for reliable phrase separation.

Features:
- Real-time microphone input
- VAD-based speech detection
- Audio level-based silence detection
- Natural phrase boundary detection
- Configurable parameters
- No complex dependencies
"""

import os
import sys
import time
import signal
import atexit
import threading
import queue
from datetime import datetime
from pathlib import Path

import numpy as np
import pyaudio

# Try to import webrtcvad-wheels first, fallback to webrtcvad
try:
    import webrtcvad_wheels as webrtcvad
    print("✓ Using webrtcvad-wheels for VAD")
except ImportError:
    try:
        import webrtcvad
        print("✓ Using webrtcvad for VAD")
    except ImportError:
        print("✗ No VAD library found. Please install webrtcvad-wheels")
        print("  pip install webrtcvad-wheels")
        sys.exit(1)

class SimpleRealtimeVoiceChunker:
    def __init__(self):
        # Audio settings
        self.SAMPLE_RATE = 44100
        self.CHUNK_SIZE = 2048
        self.CHANNELS = 1
        self.FORMAT = pyaudio.paInt16
        
        # VAD settings
        self.VAD_MODE = 2  # Aggressiveness level (0-3)
        self.VAD_SAMPLE_RATE = 16000  # VAD works better at 16kHz
        self.VAD_FRAME_DURATION = 0.03  # 30ms frames
        
        # Pause detection settings
        self.SILENCE_THRESHOLD = 0.008  # Audio level threshold for silence
        self.SILENCE_DURATION = 0.5  # Seconds of silence to end phrase
        self.MIN_PHRASE_DURATION = 1.0  # Minimum phrase duration
        self.MAX_PHRASE_DURATION = 8.0  # Maximum phrase duration
        
        # State management
        self.is_running = False
        self.audio = pyaudio.PyAudio()
        self.stream = None
        
        # Initialize VAD
        self.vad = webrtcvad.Vad(self.VAD_MODE)
        
        # Callback for when phrases are detected
        self.on_phrase_detected = None
        
        print(f"🎤 Simple Real-time Voice Chunker initialized:")
        print(f"   Sample Rate: {self.SAMPLE_RATE}Hz")
        print(f"   VAD Mode: {self.VAD_MODE}")
        print(f"   Silence Threshold: {self.SILENCE_THRESHOLD}")
        print(f"   Min Phrase Duration: {self.MIN_PHRASE_DURATION}s")
        print(f"   Max Phrase Duration: {self.MAX_PHRASE_DURATION}s")
        print(f"   Silence Duration: {self.SILENCE_DURATION}s")
    
    def set_phrase_callback(self, callback):
        """Set callback function to be called when a phrase is detected"""
        self.on_phrase_detected = callback
    
    def detect_speech_in_chunk(self, audio_chunk):
        """Detect speech in audio chunk using WebRTC VAD"""
        try:
            # Convert to 16kHz for VAD
            audio_16k = self._resample_audio(audio_chunk, self.VAD_SAMPLE_RATE)
            
            # Convert to int16
            audio_int16 = (audio_16k * 32767).astype(np.int16)
            
            # Frame size for VAD
            frame_size = int(self.VAD_SAMPLE_RATE * self.VAD_FRAME_DURATION)
            
            speech_detected = False
            
            # Check each frame
            for i in range(0, len(audio_int16) - frame_size, frame_size):
                frame = audio_int16[i:i + frame_size]
                if len(frame) == frame_size:
                    is_speech = self.vad.is_speech(frame.tobytes(), self.VAD_SAMPLE_RATE)
                    if is_speech:
                        speech_detected = True
                        break
            
            return speech_detected
            
        except Exception as e:
            print(f"❌ VAD error: {e}")
            return True  # Default to speech if VAD fails
    
    def _resample_audio(self, audio_data, target_sample_rate):
        """Simple resampling by interpolation"""
        if target_sample_rate == self.SAMPLE_RATE:
            return audio_data
        
        # Simple linear interpolation resampling
        original_length = len(audio_data)
        target_length = int(original_length * target_sample_rate / self.SAMPLE_RATE)
        
        indices = np.linspace(0, original_length - 1, target_length)
        return np.interp(indices, np.arange(original_length), audio_data)
    
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
        
        while self.is_running:
            try:
                # Read audio chunk
                data = self.stream.read(self.CHUNK_SIZE, exception_on_overflow=False)
                audio_buffer += data
                
                # Calculate audio level
                audio_data = np.frombuffer(data, dtype=np.int16)
                audio_level = np.sqrt(np.mean(audio_data.astype(np.float32) ** 2)) / 32768.0
                
                current_time = time.time()
                phrase_duration = current_time - phrase_start_time
                
                # Detect speech using VAD
                is_speech = self.detect_speech_in_chunk(audio_data)
                
                # Detect silence using audio level
                if audio_level < self.SILENCE_THRESHOLD:
                    if silence_start is None:
                        silence_start = current_time
                else:
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
        output_dir = Path("voice_chunks")
        output_dir.mkdir(exist_ok=True)
        
        # Save as WAV file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = output_dir / f"phrase_{phrase_num:02d}_{timestamp}.wav"
        
        # Convert to numpy array and save
        audio_array = np.frombuffer(audio_buffer, dtype=np.int16)
        
        # Save using soundfile (if available) or pyaudio
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

def main():
    """Main function with example usage"""
    print("🎤 Simple Real-time Voice Chunker")
    print("=" * 50)
    
    # Create chunker
    chunker = SimpleRealtimeVoiceChunker()
    
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