#!/usr/bin/env python3
"""
Real-time Microphone STS Pipeline
=================================

Combines real-time microphone chunking with parallel STS processing.
Uses ElevenLabs PRO subscription for concurrent processing.
Outputs combined MP3 file (playback commented out as requested).

Features:
- Real-time microphone input with VAD chunking
- Parallel STS processing (max 5 concurrent requests)
- Natural phrase boundary detection
- Combined MP3 output
- No real-time playback (commented out)
"""

import os
import sys
import time
import json
import queue
import signal
import asyncio
import threading
import tempfile
import argparse
import numpy as np
import requests
from pydub import AudioSegment
import pyaudio
import webrtcvad
import atexit
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Tuple, Optional
from datetime import datetime
from pathlib import Path

# Global variables for cleanup
running = True
audio_chunks = []

def signal_handler(signum, frame):
    """Handle interrupt signal"""
    global running
    print("\n⏹️  Received interrupt signal, stopping pipeline...")
    running = False

def cleanup_handler():
    """Cleanup handler for atexit"""
    global running, audio_chunks
    if running:
        running = False
        print("💾 Saving output on exit...")
        if audio_chunks:
            save_audio_chunks(audio_chunks, "realtime_mic_sts_output.mp3")

def save_audio_chunks(chunks, output_file):
    """Save audio chunks to output file"""
    if not chunks:
        print("⚠️ No audio chunks to save")
        return
    
    try:
        combined_audio = b''.join(chunks)
        with open(output_file, 'wb') as f:
            f.write(combined_audio)
        print(f"✅ All audio saved to: {output_file}")
        print(f"📊 Total audio size: {len(combined_audio)} bytes")
    except Exception as e:
        print(f"❌ Error saving audio: {e}")

# Register signal handlers
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)
atexit.register(cleanup_handler)

class RealtimeMicSTSPipeline:
    def __init__(self, output_file=None):
        # Audio settings
        self.SAMPLE_RATE = 44100
        self.CHUNK_SIZE = 2048
        self.CHANNELS = 1
        self.FORMAT = pyaudio.paInt16
        
        # VAD settings
        self.VAD_MODE = 2  # Aggressiveness level (0-3)
        self.VAD_SAMPLE_RATE = 16000  # VAD works better at 16kHz
        self.VAD_FRAME_DURATION = 0.03  # 30ms frames
        
        # Pause detection settings - More sensitive to pauses
        self.SILENCE_THRESHOLD = 0.005  # Lower threshold for silence detection (was 0.008)
        self.SILENCE_DURATION = 0.3  # Shorter silence duration to end phrase (was 0.5)
        self.MIN_PHRASE_DURATION = 0.8  # Shorter minimum phrase duration (was 1.0)
        self.MAX_PHRASE_DURATION = 6.0  # Shorter maximum phrase duration (was 8.0)
        
        # Output file
        if output_file:
            self.OUTPUT_FILE = output_file
        else:
            self.OUTPUT_FILE = f"realtime_mic_sts_output_{time.strftime('%Y%m%d_%H%M%S')}.mp3"
        
        # Audio processing
        self.audio = pyaudio.PyAudio()
        self.stream = None
        self.is_running = False
        
        # Initialize VAD
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
        
        self.vad = webrtcvad.Vad(self.VAD_MODE)
        
        # ElevenLabs PRO settings
        self.api_key = self._get_api_key()
        self.voice_id = "21m00Tcm4TlvDq8ikWAM"  # Rachel voice
        self.model_id = "eleven_multilingual_sts_v2"
        
        # Rachel voice settings
        self.voice_settings = {
            "stability": 0.5,           # 50%
            "similarity_boost": 0.37,  # 37%
            "style": 0.05,             # 5% (Exaggeration)
            "use_speaker_boost": True
        }
        
        # PRO audio settings
        self.output_format = "mp3_44100_192"
        self.optimize_streaming_latency = 3
        
        # Parallel processing settings
        self.MAX_CONCURRENT_REQUESTS = 5
        self.request_semaphore = threading.Semaphore(self.MAX_CONCURRENT_REQUESTS)
        self.order_queue = queue.Queue()  # Maintain chunk order
        self.completed_chunks: Dict[int, bytes] = {}  # Store completed chunks by index
        self.next_chunk_index = 0  # Track next chunk to play
        self.processing_complete = False  # Track when all processing is done
        
        # Performance tracking
        self.active_requests = 0
        self.completed_requests = 0
        self.failed_requests = 0
        self.total_chunks = 0
        
        # Output collection
        self.all_audio_chunks = []
        
        # Output folder for individual chunks
        self.chunks_folder = "sts_result_chunks"
        self._create_chunks_folder()
        
        print(f"🎤🎵 Real-time Microphone STS Pipeline Configuration:")
        print(f"   Output File: {self.OUTPUT_FILE}")
        print(f"   Chunks Folder: {self.chunks_folder}")
        print(f"   Voice ID: {self.voice_id}")
        print(f"   Model: {self.model_id}")
        print(f"   Max Concurrent Requests: {self.MAX_CONCURRENT_REQUESTS}")
        print(f"   Voice Settings: {self.voice_settings}")
        print(f"   Output Format: {self.output_format}")
        print(f"   VAD Mode: {self.VAD_MODE}")
        print(f"   Silence Threshold: {self.SILENCE_THRESHOLD}")
        print(f"   Min Phrase Duration: {self.MIN_PHRASE_DURATION}s")
        print(f"   Max Phrase Duration: {self.MAX_PHRASE_DURATION}s")
        print(f"   Silence Duration: {self.SILENCE_DURATION}s")
        
    def _get_api_key(self):
        """Get API key from environment or .env file"""
        api_key = os.getenv("ELEVENLABS_API_KEY")
        if api_key:
            return api_key
        
        try:
            if os.path.exists(".env"):
                with open(".env", "r") as f:
                    for line in f:
                        if line.startswith("ELEVENLABS_API_KEY="):
                            api_key = line.split("=", 1)[1].strip()
                            if api_key and api_key != "your_api_key_here":
                                return api_key
        except Exception:
            pass
        
        print("🔑 ElevenLabs API Key not found.")
        print("Please enter your ElevenLabs API key:")
        api_key = input("API Key: ").strip()
        
        if api_key:
            try:
                with open(".env", "w") as f:
                    f.write(f"ELEVENLABS_API_KEY={api_key}\n")
                print("✅ API key saved to .env file for future use")
            except Exception:
                print("⚠️ Could not save API key to file")
            
            return api_key
        
        return None
    
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
    
    async def process_chunk_with_sts_async(self, chunk_audio, chunk_index, max_retries=3):
        """Process a single chunk with ElevenLabs STS API (async version)"""
        async with self.request_semaphore:  # Limit concurrent requests
            self.active_requests += 1
            print(f"🎵 STS: Processing chunk {chunk_index} (active: {self.active_requests})")
            
            try:
                for attempt in range(max_retries):
                    try:
                        # Convert chunk to WAV for API
                        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                            chunk_audio.export(temp_file.name, format="wav")
                            temp_file_path = temp_file.name
                        
                        # Prepare the request
                        headers = {
                            "xi-api-key": self.api_key
                        }
                        
                        with open(temp_file_path, "rb") as audio_file:
                            files = {
                                "audio": ("audio.wav", audio_file, "audio/wav"),
                                "model_id": (None, self.model_id),
                                "remove_background_noise": (None, "false"),
                                "optimize_streaming_latency": (None, str(self.optimize_streaming_latency)),
                                "output_format": (None, self.output_format),
                                "voice_settings": (None, json.dumps(self.voice_settings))
                            }
                            
                            # Use asyncio to make HTTP request
                            loop = asyncio.get_event_loop()
                            response = await loop.run_in_executor(
                                None,
                                lambda: requests.post(
                                    f"https://api.elevenlabs.io/v1/speech-to-speech/{self.voice_id}/stream",
                                    headers=headers,
                                    files=files,
                                    timeout=60  # Increased timeout to 60 seconds
                                )
                            )
                        
                        # Clean up temp file
                        os.unlink(temp_file_path)
                        
                        if response.status_code == 200:
                            audio_output = response.content
                            if audio_output:
                                print(f"✅ STS: Chunk {chunk_index} completed ({len(audio_output)} bytes)")
                                self.completed_requests += 1
                                return audio_output
                            else:
                                print(f"⚠️ STS: No audio data for chunk {chunk_index}")
                        else:
                            print(f"❌ STS API error for chunk {chunk_index}: {response.status_code}")
                            
                    except Exception as e:
                        print(f"❌ STS processing error for chunk {chunk_index}: {e}")
                        if attempt < max_retries - 1:
                            await asyncio.sleep(1)
                            continue
                
                self.failed_requests += 1
                return None
                
            finally:
                self.active_requests -= 1
    
    def process_phrase_sync(self, audio_buffer, phrase_num, duration):
        """Process a phrase through STS API synchronously (for threading)"""
        try:
            # Filter out chunks less than 1 second (noise)
            if duration < 1.0:
                print(f"🎤 Phrase {phrase_num} detected: {duration:.2f}s - TOO SHORT, skipping (noise)")
                return
            
            print(f"🎤 Phrase {phrase_num} detected: {duration:.2f}s - sending to STS...")
            
            # Convert audio buffer to AudioSegment
            audio_array = np.frombuffer(audio_buffer, dtype=np.int16)
            audio_segment = AudioSegment(
                data=audio_array.tobytes(),
                sample_width=2,  # 16-bit
                frame_rate=self.SAMPLE_RATE,
                channels=1
            )
            
            # Process through STS API synchronously
            result = self.process_chunk_with_sts_sync(audio_segment, phrase_num)
            
            if result:
                # Store the result
                self.completed_chunks[phrase_num] = result
                self.all_audio_chunks.append(result)
                global audio_chunks
                audio_chunks.append(result)
                
                # Save individual chunk to file
                self.save_sts_chunk_to_file(result, phrase_num, duration)
                
                print(f"✅ Phrase {phrase_num} processed and stored")
                
            else:
                print(f"❌ Phrase {phrase_num} failed to process")
                
        except Exception as e:
            print(f"❌ Error processing phrase {phrase_num}: {e}")
    
    def process_chunk_with_sts_sync(self, chunk_audio, chunk_index, max_retries=3):
        """Process a single chunk with ElevenLabs STS API (synchronous version)"""
        with self.request_semaphore:  # Limit concurrent requests
            self.active_requests += 1
            print(f"🎵 STS: Processing chunk {chunk_index} (active: {self.active_requests})")
            
            try:
                for attempt in range(max_retries):
                    try:
                        # Convert chunk to WAV for API
                        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                            chunk_audio.export(temp_file.name, format="wav")
                            temp_file_path = temp_file.name
                        
                        # Prepare the request
                        headers = {
                            "xi-api-key": self.api_key
                        }
                        
                        with open(temp_file_path, "rb") as audio_file:
                            files = {
                                "audio": ("audio.wav", audio_file, "audio/wav"),
                                "model_id": (None, self.model_id),
                                "remove_background_noise": (None, "false"),
                                "optimize_streaming_latency": (None, str(self.optimize_streaming_latency)),
                                "output_format": (None, self.output_format),
                                "voice_settings": (None, json.dumps(self.voice_settings))
                            }
                            
                            # Make synchronous HTTP request
                            response = requests.post(
                                f"https://api.elevenlabs.io/v1/speech-to-speech/{self.voice_id}/stream",
                                headers=headers,
                                files=files,
                                timeout=60  # Increased timeout to 60 seconds
                            )
                        
                        # Clean up temp file
                        os.unlink(temp_file_path)
                        
                        if response.status_code == 200:
                            audio_output = response.content
                            if audio_output:
                                print(f"✅ STS: Chunk {chunk_index} completed ({len(audio_output)} bytes)")
                                self.completed_requests += 1
                                return audio_output
                            else:
                                print(f"⚠️ STS: No audio data for chunk {chunk_index}")
                        else:
                            print(f"❌ STS API error for chunk {chunk_index}: {response.status_code}")
                            
                    except Exception as e:
                        print(f"❌ STS processing error for chunk {chunk_index}: {e}")
                        if attempt < max_retries - 1:
                            time.sleep(1)
                            continue
                
                self.failed_requests += 1
                return None
                
            except Exception as e:
                print(f"❌ Error in STS processing for chunk {chunk_index}: {e}")
                return None
            finally:
                self.active_requests -= 1
    
    # COMMENTED OUT: Playback functionality
    # async def stream_chunk_immediately(self, chunk_index, audio_output):
    #     """Stream chunk immediately when it's ready"""
    #     try:
    #         # Add to playback buffer immediately
    #         self.continuous_buffer.put_nowait(audio_output)
    #         print(f"🔍 Chunk {chunk_index} added to playback buffer")
    #         
    #         # Add 0.5s silence after the chunk for spacing
    #         silence_duration = int(0.5 * self.SAMPLE_RATE * 2)  # 0.5s of silence (16-bit)
    #         silence_data = b'\x00' * silence_duration
    #         self.continuous_buffer.put_nowait(silence_data)
    #         print(f"🔇 Added 0.5s silence after chunk {chunk_index}")
    #         
    #         # Collect for file saving
    #         self.all_audio_chunks.append(audio_output)
    #         global audio_chunks
    #         audio_chunks.append(audio_output)
    #         
    #     except queue.Full:
    #         print(f"⚠️ Playback buffer full, skipping chunk {chunk_index}")
    #     except Exception as e:
    #         print(f"❌ Error streaming chunk {chunk_index}: {e}")
    
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
            
            # Wait for the thread to complete or until interrupted
            while self.is_running and self.listen_thread.is_alive():
                time.sleep(0.1)
                
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
        
        while self.is_running and running:  # Check both local and global running flags
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
                    self.total_chunks += 1
                    duration = len(audio_buffer) / (self.SAMPLE_RATE * 2)  # 16-bit = 2 bytes
                    
                    print(f"🎤 Phrase {phrase_count} detected: {duration:.2f}s ({end_reason})")
                    
                    # Process phrase in a separate thread for parallel processing
                    processing_thread = threading.Thread(
                        target=self.process_phrase_sync,
                        args=(audio_buffer, phrase_count, duration)
                    )
                    processing_thread.daemon = True
                    processing_thread.start()
                    
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
    
    def stop_pipeline(self):
        """Stop the entire pipeline"""
        global running
        running = False
        self.is_running = False
        print("🛑 Stopping pipeline...")
    
    def start(self):
        """Start the real-time microphone STS pipeline"""
        try:
            print("🎤🎵 Real-time Microphone STS Pipeline")
            print("=" * 80)
            print("🎤 Microphone: Real-time voice input with VAD chunking")
            print("🚀 Parallel STS: Sends chunks concurrently (max 5)")
            print("💾 Output: Combined MP3 file (playback commented out)")
            print(f"📁 Output file: {self.OUTPUT_FILE}")
            print(f"🎵 Model: {self.model_id}")
            print(f"🎵 Voice: {self.voice_id}")
            print(f"⚡ Max Concurrent Requests: {self.MAX_CONCURRENT_REQUESTS}")
            print("=" * 80)
            
            print("✅ Real-time Microphone STS Pipeline started!")
            print("🎤 Speak into microphone - phrases will be processed through STS")
            print("💾 Output will be saved to MP3 file")
            print("⏹️  Press Ctrl+C to stop")
            
            # Start listening (this will block until stopped)
            self.start_listening()
            
            # Wait for all processing to complete
            print("⏳ Waiting for all chunks to finish processing...")
            timeout = 30  # 30 second timeout
            start_time = time.time()
            while self.active_requests > 0 and (time.time() - start_time) < timeout:
                time.sleep(0.1)
            
            if self.active_requests > 0:
                print(f"⚠️ {self.active_requests} requests still processing after timeout")
            
            print("✅ Real-time Microphone STS Pipeline completed")
            
        except Exception as e:
            print(f"❌ Real-time Microphone STS Pipeline error: {e}")
        finally:
            # Always save output even on interrupt
            print("💾 Saving final output...")
            self.save_final_output()
    
    def save_final_output(self):
        """Save final output file"""
        try:
            print("💾 Saving final output...")
            save_audio_chunks(self.all_audio_chunks, self.OUTPUT_FILE)
        except Exception as e:
            print(f"❌ Error saving final output: {e}")

    def _create_chunks_folder(self):
        """Create the chunks folder if it doesn't exist"""
        try:
            if not os.path.exists(self.chunks_folder):
                os.makedirs(self.chunks_folder)
                print(f"📁 Created chunks folder: {self.chunks_folder}")
        except Exception as e:
            print(f"❌ Error creating chunks folder: {e}")
    
    def save_sts_chunk_to_file(self, audio_data, chunk_index, duration):
        """Save individual STS result chunk to file"""
        try:
            # Create filename with timestamp and chunk info
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"sts_chunk_{chunk_index:03d}_{timestamp}.mp3"
            filepath = os.path.join(self.chunks_folder, filename)
            
            # Save as MP3 file
            with open(filepath, 'wb') as f:
                f.write(audio_data)
            
            print(f"💾 Saved STS chunk {chunk_index} to: {filename} ({len(audio_data)} bytes, {duration:.2f}s)")
            return filepath
            
        except Exception as e:
            print(f"❌ Error saving STS chunk {chunk_index}: {e}")
            return None

def main():
    """Main function with improved interrupt handling"""
    global running
    sts_pipeline = None
    
    try:
        # Parse command line arguments
        parser = argparse.ArgumentParser(description="Real-time Microphone STS Pipeline")
        parser.add_argument("--output", help="Output MP3 file", default=None)
        args = parser.parse_args()
        
        # Check prerequisites
        print("✅ Prerequisites check passed")
        
        # Create and start the pipeline
        sts_pipeline = RealtimeMicSTSPipeline(args.output)
        
        # Set global running flag
        running = True
        
        # Start the real-time microphone STS pipeline
        sts_pipeline.start()
        
    except KeyboardInterrupt:
        print("\n⏹️  Interrupted by user (main)")
        running = False
        if sts_pipeline:
            sts_pipeline.stop_pipeline()
    except Exception as e:
        print(f"❌ Main error: {e}")
    finally:
        # Force cleanup
        running = False
        if sts_pipeline:
            sts_pipeline.save_final_output()
        print("🧹 Cleanup completed")

if __name__ == "__main__":
    main() 