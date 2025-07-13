#!/usr/bin/env python3
"""
Proper Streaming STS with ElevenLabs Pro Features
Uses Voice Activity Detection (VAD) for natural phrase boundaries
Streams MP3 file input → ElevenLabs STS API → Speakers
Uses Pro subscription capabilities: 192kbps MP3, advanced voice settings, optimized streaming
"""

import asyncio
import json
import base64
import time
import numpy as np
import pyaudio
import threading
import queue
from pydub import AudioSegment
import io
import os
import tempfile
import soundfile as sf
from dotenv import load_dotenv
import requests
import websockets
import argparse
import sys
import signal
import atexit

# Load environment variables
load_dotenv()

# Global variable to hold the streaming test instance
global_streaming_test = None

def signal_handler(signum, frame):
    """Handle Ctrl+C gracefully"""
    print("\n⏹️  Received interrupt signal, saving output and exiting...")
    if global_streaming_test:
        global_streaming_test.save_final_output()
    sys.exit(0)

def cleanup_handler():
    """Cleanup handler for atexit"""
    if global_streaming_test:
        print("💾 Saving output on exit...")
        global_streaming_test.save_final_output()

# Set up signal handlers
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)
atexit.register(cleanup_handler)

class ProSTSStreamingVADTest:
    def __init__(self, input_file=None):
        # Audio settings (Pro quality)
        self.FORMAT = pyaudio.paFloat32
        self.CHANNELS = 1
        self.SAMPLE_RATE = 44100  # Pro supports 44.1kHz
        self.CHUNK_SIZE = 2048  # Larger chunk for Pro
        
        # Input/Output files
        self.input_file = input_file
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.OUTPUT_FILE = f"test_pro_sts_streaming_vad_output_{timestamp}.mp3"
        
        # File logging
        self.QUEUE_LOG_FILE = "test_pro_sts_streaming_vad_log.txt"
        
        # Control
        self.running = True
        
        # STS settings (Pro optimized)
        self.STS_SAMPLE_RATE = 44100  # Pro quality
        self.STS_CHANNELS = 1
        self.STS_FORMAT = pyaudio.paInt16
        self.STS_CHUNK_SIZE = 4096  # Larger for Pro
        
        # Audio objects
        self.audio = pyaudio.PyAudio()
        self.continuous_buffer = queue.Queue(maxsize=100)
        self.playback_thread = None
        
        # Playback control
        self.playback_started = False
        self.mp3_chunks = []
        self.all_audio_chunks = []
        
        # VAD settings for natural phrase detection
        self.VAD_SAMPLE_RATE = 16000  # VAD works better at 16kHz
        self.VAD_FRAME_DURATION = 0.03  # 30ms frames
        self.VAD_FRAME_SIZE = int(self.VAD_SAMPLE_RATE * self.VAD_FRAME_DURATION)
        self.SILENCE_THRESHOLD = 0.003  # Even lower threshold for more sensitive detection
        self.MIN_PHRASE_DURATION = 0.3  # Shorter minimum to 0.3 seconds
        self.MAX_PHRASE_DURATION = 4.0  # Shorter maximum to 4 seconds
        self.SILENCE_DURATION = 0.2  # Shorter silence duration to 0.2 seconds
        self.FORCE_SPLIT_DURATION = 3.0  # Force split every 3 seconds if no silence found
        
        # ElevenLabs Pro settings
        self.api_key = self._get_api_key()
        self.voice_id = self._get_voice_id()
        self.model_id = "eleven_multilingual_sts_v2"  # Only STS model available
        
        # Pro voice settings
        self.voice_settings = {
            "stability": 0.7,  # Higher stability for Pro
            "similarity_boost": 0.8,  # Higher similarity for Pro
            "style": 0.3,  # Moderate style for natural speech
            "use_speaker_boost": True  # Pro feature
        }
        
        # Pro audio settings
        self.output_format = "mp3_44100_192"  # Pro 192kbps MP3
        self.optimize_streaming_latency = 4  # Pro streaming optimization
        
        # Timestamp for file naming
        self.timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        # Audio processing
        self.audio_segments = []
        self.current_position = 0
        
        print(f"🎵 Pro STS Streaming VAD Test Configuration:")
        print(f"   Input File: {self.input_file}")
        print(f"   Output File: {self.OUTPUT_FILE}")
        print(f"   Voice ID: {self.voice_id}")
        print(f"   Model: {self.model_id}")
        print(f"   Output Format: {self.output_format}")
        print(f"   Sample Rate: {self.STS_SAMPLE_RATE}Hz")
        print(f"   VAD Settings: Min={self.MIN_PHRASE_DURATION}s, Max={self.MAX_PHRASE_DURATION}s, Silence={self.SILENCE_DURATION}s, Force={self.FORCE_SPLIT_DURATION}s")
        print(f"   Voice Settings: {self.voice_settings}")
        
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
    
    def _get_voice_id(self):
        """Get voice ID from environment or .env file"""
        voice_id = os.getenv("ELEVENLABS_VOICE_ID")
        if voice_id:
            return voice_id
        
        try:
            if os.path.exists(".env"):
                with open(".env", "r") as f:
                    for line in f:
                        if line.startswith("ELEVENLABS_VOICE_ID="):
                            voice_id = line.split("=", 1)[1].strip()
                            if voice_id and voice_id != "your_voice_id_here":
                                return voice_id
        except Exception:
            pass
        
        # Default to Ekaterina if not found
        return "GN4wbsbejSnGSa1AzjH5"
    
    def log_to_file(self, message):
        """Log operations to file for testing"""
        try:
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            with open(self.QUEUE_LOG_FILE, 'a', encoding='utf-8') as f:
                f.write(f"[{timestamp}] {message}\n")
        except Exception as e:
            print(f"❌ Error logging to file: {e}")
    
    def load_and_prepare_audio(self):
        """Load MP3 file and prepare it for VAD processing"""
        try:
            print(f"📁 Loading audio file: {self.input_file}")
            
            # Load the MP3 file
            audio = AudioSegment.from_mp3(self.input_file)
            print(f"✅ Loaded audio: {len(audio)}ms ({len(audio)/1000:.1f}s)")
            
            # Convert to mono if stereo
            if audio.channels > 1:
                audio = audio.set_channels(1)
                print("🔄 Converted to mono")
            
            # Resample to 44.1kHz for STS processing
            if audio.frame_rate != self.STS_SAMPLE_RATE:
                audio = audio.set_frame_rate(self.STS_SAMPLE_RATE)
                print(f"🔄 Resampled to {self.STS_SAMPLE_RATE}Hz")
            
            self.audio_segments = [audio]
            print(f"✅ Audio prepared for VAD processing")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading audio file: {e}")
            return False
    
    def detect_phrase_boundaries(self, audio_segment):
        """Detect natural phrase boundaries using VAD"""
        try:
            print("🎤 VAD: Detecting natural phrase boundaries...")
            
            # Convert to 16kHz for VAD processing
            vad_audio = audio_segment.set_frame_rate(self.VAD_SAMPLE_RATE)
            
            # Convert to numpy array
            samples = np.array(vad_audio.get_array_of_samples())
            
            # Normalize audio
            if len(samples) > 0:
                samples = samples.astype(np.float32) / np.max(np.abs(samples))
            
            # Process in frames
            frame_size = self.VAD_FRAME_SIZE
            frames = []
            for i in range(0, len(samples), frame_size):
                frame = samples[i:i + frame_size]
                if len(frame) == frame_size:  # Only process complete frames
                    frames.append(frame)
            
            # Detect speech activity
            speech_frames = []
            silence_frames = []
            current_phrase_start = None
            phrases = []
            last_phrase_end = 0
            
            for i, frame in enumerate(frames):
                # Calculate frame energy
                energy = np.sqrt(np.mean(frame ** 2))
                
                # Determine if frame contains speech
                is_speech = energy > self.SILENCE_THRESHOLD
                
                if is_speech:
                    if current_phrase_start is None:
                        current_phrase_start = i * frame_size
                    speech_frames.append(i)
                else:
                    silence_frames.append(i)
                
                # Check for phrase boundaries
                if current_phrase_start is not None and len(speech_frames) > 0:
                    current_phrase_duration = (i - speech_frames[0]) * self.VAD_FRAME_DURATION
                    
                    # Force split if phrase is getting too long (even without silence)
                    if current_phrase_duration >= self.FORCE_SPLIT_DURATION:
                        # Force end current phrase
                        phrase_end = current_phrase_start + (len(speech_frames) * frame_size)
                        phrases.append((current_phrase_start, phrase_end))
                        last_phrase_end = phrase_end
                        current_phrase_start = None
                        speech_frames = []
                        silence_frames = []
                        continue
                    
                    # Check if we've had enough silence to end a phrase naturally
                    if not is_speech:
                        silence_duration = len(silence_frames) * self.VAD_FRAME_DURATION
                        phrase_duration = (i - speech_frames[0]) * self.VAD_FRAME_DURATION
                        
                        if silence_duration >= self.SILENCE_DURATION and phrase_duration >= self.MIN_PHRASE_DURATION:
                            # End current phrase
                            phrase_end = current_phrase_start + (len(speech_frames) * frame_size)
                            phrases.append((current_phrase_start, phrase_end))
                            last_phrase_end = phrase_end
                            current_phrase_start = None
                            speech_frames = []
                            silence_frames = []
            
            # Handle last phrase if still active
            if current_phrase_start is not None and len(speech_frames) > 0:
                phrase_duration = len(speech_frames) * self.VAD_FRAME_DURATION
                if phrase_duration >= self.MIN_PHRASE_DURATION:
                    phrase_end = current_phrase_start + (len(speech_frames) * frame_size)
                    phrases.append((current_phrase_start, phrase_end))
            
            # Convert frame positions back to original sample rate
            scale_factor = self.STS_SAMPLE_RATE / self.VAD_SAMPLE_RATE
            scaled_phrases = []
            for start, end in phrases:
                scaled_start = int(start * scale_factor)
                scaled_end = int(end * scale_factor)
                scaled_phrases.append((scaled_start, scaled_end))
            
            print(f"✅ VAD: Detected {len(scaled_phrases)} natural phrases")
            return scaled_phrases
            
        except Exception as e:
            print(f"❌ VAD error: {e}")
            # Fallback to fixed chunking
            return self.fallback_chunking(audio_segment)
    
    def fallback_chunking(self, audio_segment):
        """Fallback to fixed chunking if VAD fails"""
        print("⚠️ Using fallback chunking with smaller chunks...")
        
        duration_ms = len(audio_segment)
        chunk_duration_ms = 2000  # 2 seconds (even smaller chunks for better processing)
        phrases = []
        
        for i in range(0, duration_ms, chunk_duration_ms):
            start = i
            end = min(i + chunk_duration_ms, duration_ms)
            phrases.append((start, end))
        
        print(f"✅ Fallback: Created {len(phrases)} chunks of {chunk_duration_ms}ms each")
        return phrases
    
    def stream_audio_by_phrases(self):
        """Stream audio using natural phrase boundaries"""
        print("🎤 Pro STS Streaming VAD: Starting phrase-based streaming...")
        
        if not self.audio_segments:
            print("❌ No audio loaded")
            return
        
        audio = self.audio_segments[0]
        
        # Detect phrase boundaries
        phrases = self.detect_phrase_boundaries(audio)
        
        if not phrases:
            print("❌ No phrases detected, using fallback chunking...")
            phrases = self.fallback_chunking(audio)
        
        # Check if we have enough valid phrases
        valid_phrases = []
        for start_ms, end_ms in phrases:
            phrase_duration = (end_ms - start_ms) / 1000.0
            if self.MIN_PHRASE_DURATION <= phrase_duration <= self.MAX_PHRASE_DURATION:
                valid_phrases.append((start_ms, end_ms))
        
        # If too few valid phrases, use fallback
        if len(valid_phrases) < 3:  # Need at least 3 valid phrases
            print(f"⚠️ Only {len(valid_phrases)} valid phrases detected, using fallback chunking...")
            phrases = self.fallback_chunking(audio)
            valid_phrases = phrases  # All fallback chunks are valid
        
        chunk_count = 0
        
        for i, (start_ms, end_ms) in enumerate(phrases):
            if not self.running:
                break
                
            chunk_count += 1
            phrase_duration = (end_ms - start_ms) / 1000.0
            
            print(f"🎵 Pro STS Streaming VAD: Processing phrase {chunk_count}/{len(phrases)} ({phrase_duration:.1f}s)")
            
            # Extract phrase
            phrase_audio = audio[start_ms:end_ms]
            
            # Skip phrases that are too short
            if phrase_duration < self.MIN_PHRASE_DURATION:
                print(f"⚠️ Skipping phrase {chunk_count} - too short ({phrase_duration:.1f}s)")
                continue
            
            # Skip phrases that are too long
            if phrase_duration > self.MAX_PHRASE_DURATION:
                print(f"⚠️ Skipping phrase {chunk_count} - too long ({phrase_duration:.1f}s)")
                continue
            
            # Convert phrase to bytes
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                phrase_audio.export(temp_file.name, format="wav")
                temp_file_path = temp_file.name
            
            # Read the audio file
            with open(temp_file_path, "rb") as audio_file:
                audio_data = audio_file.read()
            
            # Clean up temp file
            os.unlink(temp_file_path)
            
            # Process with STS API
            success = self.process_audio_with_sts_pro(audio_data, chunk_count)
            
            if not success:
                print(f"❌ Failed to process phrase {chunk_count}, continuing...")
            
            # Small delay between phrases for smooth streaming
            time.sleep(0.1)
        
        print(f"🎵 Pro STS Streaming VAD: Processed {chunk_count} phrases")
    
    def process_audio_with_sts_pro(self, audio_data, chunk_index):
        """Process audio using ElevenLabs Speech-to-Speech API with Pro features"""
        try:
            print(f"🎵 Pro STS: Sending phrase {chunk_index} to ElevenLabs STS API with Pro features...")
            
            # Convert bytes to numpy array
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
            
            # Save audio to temporary file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                sf.write(temp_file, audio_array, self.STS_SAMPLE_RATE)
                temp_file_path = temp_file.name
            
            # Prepare the request with Pro features
            headers = {
                "xi-api-key": self.api_key
            }
            
            # Read the audio file
            with open(temp_file_path, "rb") as audio_file:
                files = {
                    "audio": ("audio.wav", audio_file, "audio/wav"),
                    "model_id": (None, self.model_id),
                    "remove_background_noise": (None, "false"),  # Disable for better quality
                    "optimize_streaming_latency": (None, str(self.optimize_streaming_latency)),  # Pro streaming optimization
                    "output_format": (None, self.output_format),  # Pro 192kbps MP3
                    "voice_settings": (None, json.dumps(self.voice_settings))  # Pro voice settings
                }
                
                print(f"🎵 Pro STS: Sending phrase {chunk_index} to ElevenLabs STS API")
                print(f"   Model: {self.model_id}")
                print(f"   Voice: {self.voice_id}")
                print(f"   Output Format: {self.output_format}")
                print(f"   Voice Settings: {self.voice_settings}")
                
                response = requests.post(
                    f"https://api.elevenlabs.io/v1/speech-to-speech/{self.voice_id}/stream",
                    headers=headers,
                    files=files,
                    timeout=30  # 30 second timeout
                )
            
            # Clean up temp file
            os.unlink(temp_file_path)
            
            print(f"🎵 Pro STS API response: status={response.status_code}")
            
            if response.status_code == 200:
                # Get the audio data
                audio_output = response.content
                
                if audio_output:
                    print(f"✅ Pro STS: Received {len(audio_output)} bytes for phrase {chunk_index}")
                    
                    # Add to continuous buffer for playback
                    try:
                        self.continuous_buffer.put_nowait(audio_output)
                    except queue.Full:
                        # Buffer full, skip this chunk to maintain real-time
                        pass
                    
                    # Collect for file saving
                    self.mp3_chunks.append(audio_output)
                    self.all_audio_chunks.append(audio_output)
                    
                    # Save individual chunk immediately
                    chunk_filename = f"pro_sts_vad_phrase_{chunk_index}_{self.timestamp}.mp3"
                    with open(chunk_filename, 'wb') as f:
                        f.write(audio_output)
                    print(f"💾 Saved Pro STS VAD phrase {chunk_index}: {chunk_filename} ({len(audio_output)} bytes)")
                    
                    # Start playback when we have enough data
                    if not self.playback_started and len(self.mp3_chunks) >= 2:
                        self.playback_started = True
                        print(f"🎵 Starting Pro STS VAD streaming playback")
                    
                    # Log to file
                    self.log_to_file(f"PRO_STS_VAD_SUCCESS: Phrase {chunk_index}, {len(audio_output)} bytes")
                    return True
                else:
                    print(f"⚠️ Pro STS: No audio data received for phrase {chunk_index}")
                    self.log_to_file(f"PRO_STS_VAD_ERROR: Phrase {chunk_index} - No audio data received")
                    return False
            else:
                print(f"❌ Pro STS API error for phrase {chunk_index}: {response.status_code} - {response.text}")
                self.log_to_file(f"PRO_STS_VAD_ERROR: Phrase {chunk_index} - {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            print(f"❌ Pro STS processing error for phrase {chunk_index}: {e}")
            self.log_to_file(f"PRO_STS_VAD_ERROR: Phrase {chunk_index} - {e}")
            return False
    
    async def _smooth_audio_streaming(self):
        """Smooth audio streaming with continuous playback"""
        try:
            # Initialize audio stream
            stream = self.audio.open(
                format=self.FORMAT,
                channels=self.CHANNELS,
                rate=self.SAMPLE_RATE,
                output=True,
                frames_per_buffer=self.CHUNK_SIZE
            )
            
            # Start continuous playback thread if not already started
            if not self.playback_thread or not self.playback_thread.is_alive():
                self.playback_thread = threading.Thread(target=self._continuous_playback_worker, args=(stream,))
                self.playback_thread.daemon = True
                self.playback_thread.start()
            
            print("🎵 Pro STS VAD Streaming: Audio streaming initialized")
            
        except Exception as e:
            print(f"❌ Smooth streaming error: {e}")
    
    def _continuous_playback_worker(self, stream):
        """Continuous playback worker that eliminates gaps"""
        try:
            while self.running:
                try:
                    # Get audio chunk (blocking with timeout)
                    audio_data = self.continuous_buffer.get(timeout=0.1)
                    
                    # Decode and play immediately
                    self._play_audio_chunk_smooth(audio_data, stream)
                    
                except queue.Empty:
                    # No audio data, continue waiting
                    continue
                except Exception as e:
                    print(f"⚠️ Playback error: {e}")
                    continue
            
            print("🎵 Pro STS VAD streaming playback worker completed")
            
        except Exception as e:
            print(f"❌ Continuous playback worker error: {e}")
    
    def _play_audio_chunk_smooth(self, mp3_data, stream):
        """Play audio chunk with minimal processing for smooth playback"""
        try:
            if len(mp3_data) < 100:
                return
            
            try:
                audio_segment = AudioSegment.from_file(io.BytesIO(mp3_data), format="mp3")
            except Exception:
                return
            
            if len(audio_segment) == 0:
                return
            
            pcm_data = audio_segment.get_array_of_samples()
            pcm_float = np.array(pcm_data, dtype=np.float32) / 32768.0
            
            # Play immediately without additional delays
            stream.write(pcm_float.astype(np.float32).tobytes())
            
        except Exception as e:
            # Silent error handling to avoid gaps
            pass
    
    async def start(self):
        """Start the Pro STS VAD streaming test"""
        print("🎤🎵 Pro STS VAD Streaming Test: MP3 File → ElevenLabs STS API → Speakers")
        print("=" * 70)
        print("🎤 STS: Uses VAD to detect natural phrases → sends to ElevenLabs STS API")
        print("🎵 TTS: Returns converted audio → smooth live playback")
        print(f"📁 Queue log: {self.QUEUE_LOG_FILE}")
        print(f"🎵 Output file: {self.OUTPUT_FILE}")
        print(f"🎵 Model: {self.model_id}")
        print(f"🎵 Voice: {self.voice_id}")
        print(f"🎵 Pro Features: {self.output_format}, {self.voice_settings}")
        print("=" * 70)
        
        # Check if input file exists
        if not self.input_file or not os.path.exists(self.input_file):
            print(f"❌ Input file not found: {self.input_file}")
            print("   Please provide a valid MP3 file path")
            return
        
        # Clear log file
        if os.path.exists(self.QUEUE_LOG_FILE):
            os.remove(self.QUEUE_LOG_FILE)
        
        # Load and prepare audio
        if not self.load_and_prepare_audio():
            print("❌ Failed to load audio file")
            return
        
        # Start audio streaming
        await self._smooth_audio_streaming()
        
        # Start streaming thread
        streaming_thread = threading.Thread(target=self.stream_audio_by_phrases)
        streaming_thread.daemon = True
        streaming_thread.start()
        
        print("✅ Pro STS VAD streaming test started!")
        print("🎵 Streaming MP3 file using natural phrase boundaries...")
        print("🎵 Converted audio will play through speakers!")
        print("⏹️  Press Ctrl+C to stop early")
        
        try:
            # Keep main thread alive until streaming completes
            while self.running:
                await asyncio.sleep(0.1)
        except KeyboardInterrupt:
            print("\n⏹️  Stopping Pro STS VAD streaming test...")
            self.running = False
        except Exception as e:
            print(f"\n❌ Error in main loop: {e}")
            self.running = False
        finally:
            # Always save output, even if interrupted
            if self.all_audio_chunks:
                print("💾 Saving final output...")
                self.save_final_output()
                print("✅ Pro STS VAD streaming test completed")
            else:
                print("⚠️ No audio chunks processed, nothing to save")
                print("✅ Pro STS VAD streaming test completed")
    
    def save_final_output(self):
        """Save all TTS audio to a single MP3 file"""
        try:
            if self.all_audio_chunks:
                print(f"💾 Saving {len(self.all_audio_chunks)} audio chunks to {self.OUTPUT_FILE}...")
                
                # Combine all audio chunks
                combined_audio = b''.join(self.all_audio_chunks)
                
                # Save as MP3
                with open(self.OUTPUT_FILE, 'wb') as f:
                    f.write(combined_audio)
                
                print(f"✅ All audio saved to: {self.OUTPUT_FILE}")
                print(f"📊 Total audio size: {len(combined_audio)} bytes")
                print(f"🎵 Duration: {len(combined_audio) / (192 * 1024 / 8) * 1000:.1f} seconds")
                
                # Log final stats
                self.log_to_file(f"FINAL: Saved {len(self.all_audio_chunks)} chunks, {len(combined_audio)} bytes")
            else:
                print("⚠️ No audio chunks to save")
                self.log_to_file("FINAL: No audio chunks to save")
                
        except Exception as e:
            print(f"❌ Error saving final output: {e}")
            self.log_to_file(f"ERROR: Failed to save final output - {e}")

async def main():
    """Main function"""
    global global_streaming_test
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Pro STS VAD Streaming Test - Stream MP3 file with natural phrase boundaries')
    parser.add_argument('input_file', help='Input MP3 file path')
    parser.add_argument('--output', '-o', help='Output file path (optional)')
    
    args = parser.parse_args()
    
    # Check prerequisites
    if not os.getenv("ELEVENLABS_API_KEY"):
        print("❌ ELEVENLABS_API_KEY not found")
        print("   Please set your ElevenLabs API key in environment or .env file")
        return
    
    print("✅ Prerequisites check passed")
    
    # Create and start Pro STS VAD streaming test
    pro_sts_vad_streaming = ProSTSStreamingVADTest(args.input_file)
    global_streaming_test = pro_sts_vad_streaming  # Assign to global variable
    
    # Override output file name if specified
    if args.output:
        pro_sts_vad_streaming.OUTPUT_FILE = args.output
        print(f"🎵 Output file set to: {pro_sts_vad_streaming.OUTPUT_FILE}")
    
    try:
        await pro_sts_vad_streaming.start()
    except KeyboardInterrupt:
        print("\n⏹️  Interrupted by user, saving output...")
        pro_sts_vad_streaming.save_final_output()
        print("✅ Output saved successfully")
    except Exception as e:
        print(f"\n❌ Error during processing: {e}")
        print("💾 Attempting to save any processed chunks...")
        pro_sts_vad_streaming.save_final_output()
    finally:
        # Ensure output is saved even if no exception was caught
        if pro_sts_vad_streaming.all_audio_chunks:
            print("💾 Final cleanup - saving output...")
            pro_sts_vad_streaming.save_final_output()

if __name__ == "__main__":
    asyncio.run(main()) 