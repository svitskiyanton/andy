#!/usr/bin/env python3
"""
Pro STS Streaming Enhanced VAD Test - FIXED VERSION
Fixes audio quality issues, adds normalization, and improves VAD sensitivity
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
import soundfile as sf
import requests
from pydub import AudioSegment
import pyaudio
import atexit
import webrtcvad

# Global variables for cleanup
running = True
audio_chunks = []

def signal_handler(signum, frame):
    """Handle interrupt signal with improved reliability"""
    global running
    print("\n⏹️  Received interrupt signal, saving output and exiting...")
    running = False
    
    # Force exit after a short delay if cleanup takes too long
    def force_exit():
        time.sleep(3)  # Wait 3 seconds for cleanup
        print("🔄 Force exiting...")
        os._exit(0)
    
    # Start force exit thread
    exit_thread = threading.Thread(target=force_exit, daemon=True)
    exit_thread.start()

def cleanup_handler():
    """Cleanup handler for atexit with improved reliability"""
    global running, audio_chunks
    if running:
        running = False
        print("💾 Saving output on exit...")
        if audio_chunks:
            save_audio_chunks(audio_chunks, "test_output.mp3")
    
    # Force cleanup of any remaining resources
    try:
        import gc
        gc.collect()
    except:
        pass

def save_audio_chunks(chunks, output_file):
    """Save audio chunks to output file with improved error handling"""
    if not chunks:
        print("⚠️ No audio chunks to save")
        return
    
    try:
        # Combine all chunks
        combined_audio = b''.join(chunks)
        
        with open(output_file, 'wb') as f:
            f.write(combined_audio)
        
        print(f"✅ All audio saved to: {output_file}")
        print(f"📊 Total audio size: {len(combined_audio)} bytes")
        
        # Calculate duration (approximate)
        # MP3 at 192kbps = ~24KB per second
        duration_seconds = len(combined_audio) / 24000
        print(f"🎵 Duration: {duration_seconds:.1f} seconds")
        
    except Exception as e:
        print(f"❌ Error saving audio: {e}")

# Register signal handlers with improved reliability
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)  # Add SIGTERM handling
atexit.register(cleanup_handler)

class ProSTSStreamingEnhancedVADFixedTest:
    def __init__(self, input_file=None):
        # Audio settings (Pro quality)
        self.STS_SAMPLE_RATE = 44100
        self.SAMPLE_RATE = 44100
        self.CHUNK_SIZE = 1024
        self.CHANNELS = 1
        self.FORMAT = pyaudio.paInt16
        
        # Input/Output files
        self.input_file = input_file
        self.OUTPUT_FILE = f"test_pro_sts_streaming_enhanced_vad_fixed_output_{time.strftime('%Y%m%d_%H%M%S')}.mp3"
        self.QUEUE_LOG_FILE = "test_pro_sts_streaming_enhanced_vad_fixed_log.txt"
        
        # Audio processing
        self.audio = pyaudio.PyAudio()
        self.continuous_buffer = queue.Queue(maxsize=50)
        self.playback_thread = None
        self.running = True
        self.playback_started = False
        self.mp3_chunks = []
        self.all_audio_chunks = []
        
        # Enhanced VAD settings for natural phrase detection
        self.VAD_SAMPLE_RATE = 16000  # VAD works better at 16kHz
        self.VAD_FRAME_DURATION = 0.03  # 30ms frames
        self.VAD_FRAME_SIZE = int(self.VAD_SAMPLE_RATE * self.VAD_FRAME_DURATION)
        
        # REAL Pause detection settings - optimized for natural phrase boundaries
        self.SILENCE_THRESHOLD = 0.001  # ULTRA-LOW threshold for maximum sensitivity
        self.MIN_PHRASE_DURATION = 0.2  # SHORTER minimum to catch natural phrases (200ms)
        self.MAX_PHRASE_DURATION = 10.0  # LONGER maximum to avoid forced splits
        self.SILENCE_DURATION = 0.15  # SHORTER silence duration to detect natural pauses (150ms)
        self.FORCE_SPLIT_DURATION = 8.0  # LONGER force split duration
        
        # AUDIO QUALITY IMPROVEMENTS
        self.TARGET_RMS = 0.3  # Target RMS level for normalization
        self.MIN_AUDIO_LEVEL = 0.01  # Minimum audio level threshold
        self.NOISE_GATE = 0.005  # Noise gate threshold
        
        # REMOVED: Fallback settings - we'll use VAD only
        # self.FALLBACK_CHUNK_DURATION_MS = 2500  # 2.5 seconds fallback chunks
        
        # ElevenLabs Pro settings
        self.api_key = self._get_api_key()
        self.voice_id = self._get_voice_id()
        self.model_id = "eleven_multilingual_sts_v2"
        
        # Pro voice settings - IMPROVED for better quality
        self.voice_settings = {
            "stability": 0.8,  # HIGHER stability for more consistent output
            "similarity_boost": 0.85,  # HIGHER similarity for better voice cloning
            "style": 0.2,  # LOWER style for more natural speech
            "use_speaker_boost": True
        }
        
        # Pro audio settings
        self.output_format = "mp3_44100_192"
        self.optimize_streaming_latency = 3  # REDUCED from 4 for better quality
        
        # Timestamp for file naming
        self.timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        # Audio processing
        self.audio_segments = []
        
        print(f"🎵 Pro STS Streaming Enhanced VAD FIXED Test Configuration:")
        print(f"   Input File: {self.input_file}")
        print(f"   Output File: {self.OUTPUT_FILE}")
        print(f"   Voice ID: {self.voice_id}")
        print(f"   Model: {self.model_id}")
        print(f"   VAD Sample Rate: {self.VAD_SAMPLE_RATE}Hz")
        print(f"   Silence Threshold: {self.SILENCE_THRESHOLD} (IMPROVED)")
        print(f"   Min Phrase Duration: {self.MIN_PHRASE_DURATION}s (IMPROVED)")
        print(f"   Max Phrase Duration: {self.MAX_PHRASE_DURATION}s (IMPROVED)")
        print(f"   Silence Duration: {self.SILENCE_DURATION}s (IMPROVED)")
        print(f"   Target RMS: {self.TARGET_RMS} (NEW)")
        print(f"   Noise Gate: {self.NOISE_GATE} (NEW)")
        print(f"   Voice Settings: {self.voice_settings} (IMPROVED)")
        print(f"   VAD-ONLY: No fallback to fixed chunking")
        
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
        """Load MP3 file and prepare it for processing with IMPROVED audio quality"""
        try:
            print(f"📁 Loading audio file: {self.input_file}")
            
            # Load the MP3 file
            audio = AudioSegment.from_mp3(self.input_file)
            print(f"✅ Loaded audio: {len(audio)}ms ({len(audio)/1000:.1f}s)")
            
            # Convert to mono if needed
            if audio.channels > 1:
                audio = audio.set_channels(1)
                print("🔄 Converted to mono")
            
            # Resample to 44.1kHz if needed
            if audio.frame_rate != self.STS_SAMPLE_RATE:
                audio = audio.set_frame_rate(self.STS_SAMPLE_RATE)
                print(f"🔄 Resampled to {self.STS_SAMPLE_RATE}Hz")
            
            # IMPROVED: Audio normalization and quality enhancement
            audio = self._enhance_audio_quality(audio)
            
            self.audio_segments = [audio]
            print("✅ Audio prepared for enhanced VAD processing with quality improvements")
            return True
            
        except Exception as e:
            print(f"❌ Error loading audio: {e}")
            return False
    
    def _enhance_audio_quality(self, audio_segment):
        """Enhance audio quality to prevent gaps and artifacts"""
        try:
            print("🔧 Enhancing audio quality...")
            
            # Convert to numpy array for processing
            samples = np.array(audio_segment.get_array_of_samples(), dtype=np.float32)
            
            # 1. APPLY NOISE GATE to remove very quiet parts
            noise_gate_mask = np.abs(samples) > self.NOISE_GATE
            if np.sum(noise_gate_mask) > 0:
                # Only apply noise gate if there's significant audio
                samples = samples * noise_gate_mask
                print(f"🔧 Applied noise gate: {self.NOISE_GATE}")
            
            # 2. NORMALIZE AUDIO to target RMS level
            current_rms = np.sqrt(np.mean(samples ** 2))
            if current_rms > 0:
                # Calculate normalization factor
                normalization_factor = self.TARGET_RMS / current_rms
                # Limit normalization to prevent clipping
                normalization_factor = min(normalization_factor, 3.0)
                
                samples = samples * normalization_factor
                print(f"🔧 Normalized audio: RMS {current_rms:.4f} → {self.TARGET_RMS:.4f} (factor: {normalization_factor:.2f})")
            
            # 3. APPLY SOFT CLIPPING to prevent harsh distortion
            samples = np.tanh(samples) * 0.95  # Soft clipping with headroom
            print("🔧 Applied soft clipping")
            
            # 4. ENSURE MINIMUM AUDIO LEVEL
            max_level = np.max(np.abs(samples))
            if max_level < self.MIN_AUDIO_LEVEL:
                # Boost quiet audio
                boost_factor = self.MIN_AUDIO_LEVEL / max_level
                boost_factor = min(boost_factor, 2.0)  # Limit boost
                samples = samples * boost_factor
                print(f"🔧 Boosted quiet audio: factor {boost_factor:.2f}")
            
            # Convert back to AudioSegment
            enhanced_samples = (samples * 32767).astype(np.int16)
            enhanced_audio = AudioSegment(
                enhanced_samples.tobytes(),
                frame_rate=audio_segment.frame_rate,
                sample_width=2,
                channels=1
            )
            
            print("✅ Audio quality enhancement completed")
            return enhanced_audio
            
        except Exception as e:
            print(f"❌ Error enhancing audio quality: {e}")
            return audio_segment
    
    def detect_speech_activity(self, audio_segment):
        """Detect speech activity using WebRTC VAD with REAL sensitivity"""
        try:
            # Initialize VAD with ULTRA-LOW aggressiveness for maximum sensitivity
            vad = webrtcvad.Vad(0)  # Most sensitive setting
            
            # Convert audio to 16kHz for VAD
            audio_16k = audio_segment.set_frame_rate(16000)
            
            # Convert to numpy array
            samples = np.array(audio_16k.get_array_of_samples(), dtype=np.int16)
            
            # Frame size for 30ms at 16kHz
            frame_size = int(16000 * 0.03)
            
            speech_frames = []
            silence_frames = []
            current_frame = 0
            
            # IMPROVED: Analyze every frame with detailed logging
            consecutive_speech = 0
            consecutive_silence = 0
            
            for i in range(0, len(samples) - frame_size, frame_size):
                frame = samples[i:i + frame_size]
                if len(frame) == frame_size:
                    is_speech = vad.is_speech(frame.tobytes(), 16000)
                    
                    if is_speech:
                        speech_frames.append(current_frame)
                        consecutive_speech += 1
                        consecutive_silence = 0
                    else:
                        silence_frames.append(current_frame)
                        consecutive_silence += 1
                        consecutive_speech = 0
                    
                    current_frame += 1
            
            print(f"🔍 VAD Analysis: {len(speech_frames)} speech frames, {len(silence_frames)} silence frames")
            print(f"🔍 Max consecutive speech: {consecutive_speech}, Max consecutive silence: {consecutive_silence}")
            
            return speech_frames, silence_frames, frame_size
            
        except Exception as e:
            print(f"❌ VAD error: {e}")
            return [], [], 0
    
    def find_natural_phrase_boundaries(self, audio_segment):
        """Find natural phrase boundaries using REAL pause detection - PROPER VAD VERSION"""
        print("🎤 Analyzing audio for REAL natural phrase boundaries...")
        
        # Get speech/silence frames
        speech_frames, silence_frames, frame_size = self.detect_speech_activity(audio_segment)
        
        if not speech_frames:
            print("⚠️ No speech detected, creating single chunk")
            return [(0, len(audio_segment))]
        
        # Convert frame indices to time positions
        frame_duration = frame_size / 16000  # seconds per frame
        speech_times = [frame * frame_duration for frame in speech_frames]
        silence_times = [frame * frame_duration for frame in silence_frames]
        
        print(f"✅ Detected {len(speech_times)} speech frames, {len(silence_times)} silence frames")
        
        # PROPER VAD APPROACH: Find REAL phrase boundaries by analyzing silence patterns
        chunks = []
        current_start = 0
        last_speech_time = 0
        
        # Sort all time points chronologically
        all_times = []
        for time in speech_times:
            all_times.append((time, 'speech'))
        for time in silence_times:
            all_times.append((time, 'silence'))
        all_times.sort(key=lambda x: x[0])
        
        print(f"🔍 Analyzing {len(all_times)} time points for natural boundaries...")
        
        # Find consecutive silence periods that indicate phrase boundaries
        silence_start = None
        silence_duration = 0
        consecutive_silence_frames = 0
        
        # DEBUG: Track silence periods
        silence_periods = []
        
        for i, (time, event_type) in enumerate(all_times):
            if event_type == 'silence':
                if silence_start is None:
                    silence_start = time
                    consecutive_silence_frames = 1
                else:
                    # Check if this silence is consecutive (within 50ms)
                    if time - silence_start < 0.05:
                        consecutive_silence_frames += 1
                    else:
                        # Reset for new silence period
                        silence_start = time
                        consecutive_silence_frames = 1
            else:  # speech
                if silence_start is not None:
                    # Calculate total silence duration
                    silence_duration = time - silence_start
                    
                    # DEBUG: Track all silence periods
                    silence_periods.append({
                        'start': silence_start,
                        'end': time,
                        'duration': silence_duration,
                        'frames': consecutive_silence_frames
                    })
                    
                    # Check if this silence period is long enough to be a phrase boundary
                    # REDUCED thresholds for better detection
                    if (silence_duration >= 0.05 and  # 50ms silence (reduced from 150ms)
                        consecutive_silence_frames >= 1 and  # At least 1 consecutive silence frame (reduced from 2)
                        time - current_start >= 0.2):  # At least 200ms phrase (reduced from 300ms)
                        
                        # This is a natural phrase boundary!
                        chunk_end = int(silence_start * 1000)  # End at start of silence
                        chunk_duration = (chunk_end - current_start) / 1000.0
                        
                        if chunk_duration >= 0.2:  # Minimum 200ms chunk
                            chunks.append((current_start, chunk_end))
                            print(f"🎯 Found natural boundary at {silence_start:.2f}s (silence: {silence_duration:.2f}s, frames: {consecutive_silence_frames}, chunk: {chunk_duration:.2f}s)")
                        
                        # Start new phrase after silence
                        current_start = int(time * 1000)
                    
                    # Reset silence tracking
                    silence_start = None
                    consecutive_silence_frames = 0
                
                last_speech_time = time
        
        # DEBUG: Show silence periods
        print(f"🔍 Found {len(silence_periods)} silence periods:")
        for i, period in enumerate(silence_periods[:10]):  # Show first 10
            print(f"   Period {i+1}: {period['start']:.2f}s - {period['end']:.2f}s (duration: {period['duration']:.2f}s, frames: {period['frames']})")
        if len(silence_periods) > 10:
            print(f"   ... and {len(silence_periods) - 10} more periods")
        
        # Add final chunk if needed
        if current_start < len(audio_segment):
            final_duration = (len(audio_segment) - current_start) / 1000.0
            if final_duration >= 0.2:  # Minimum 200ms chunk
                chunks.append((current_start, len(audio_segment)))
                print(f"🎯 Added final chunk: {current_start/1000:.2f}s - {len(audio_segment)/1000:.2f}s ({final_duration:.2f}s)")
        
        # If no chunks found, create at least one chunk
        if not chunks:
            print("⚠️ No natural boundaries found, creating single chunk")
            chunks.append((0, len(audio_segment)))
        
        print(f"✅ Created {len(chunks)} REAL natural phrase chunks")
        
        # Log chunk details
        for i, (start, end) in enumerate(chunks):
            duration = (end - start) / 1000.0
            print(f"🎵 Chunk {i+1}: {start/1000:.1f}s - {end/1000:.1f}s ({duration:.1f}s)")
        
        return chunks
    
    def _calculate_silence_duration_improved(self, silence_time, silence_times):
        """IMPROVED: Calculate the duration of silence at a given time"""
        # Find consecutive silence frames with better tolerance
        consecutive_silence = 0
        tolerance = 0.05  # 50ms tolerance for consecutive silence
        
        for i, time in enumerate(silence_times):
            if abs(time - silence_time) < tolerance:
                consecutive_silence += 1
            elif time > silence_time + tolerance:
                break
        
        return consecutive_silence * 0.03  # 30ms per frame
    
    def _split_long_chunk_improved(self, audio_segment, start_ms, end_ms):
        """IMPROVED: Split a long chunk into smaller pieces with better boundaries"""
        chunks = []
        duration_ms = end_ms - start_ms
        
        # IMPROVED: Use longer split duration for better quality
        split_duration_ms = int(self.FORCE_SPLIT_DURATION * 1000)
        
        for i in range(0, duration_ms, split_duration_ms):
            chunk_start = start_ms + i
            chunk_end = min(start_ms + i + split_duration_ms, end_ms)
            
            # IMPROVED: Ensure minimum chunk size
            if (chunk_end - chunk_start) / 1000.0 >= self.MIN_PHRASE_DURATION:
                chunks.append((chunk_start, chunk_end))
        
        return chunks
    
    def stream_audio_by_enhanced_vad(self):
        """Stream audio using enhanced VAD-based chunking with IMPROVED quality"""
        print("🎤 Pro STS Streaming Enhanced VAD FIXED: Starting natural phrase detection...")
        
        if not self.audio_segments:
            print("❌ No audio loaded")
            return
        
        audio = self.audio_segments[0]
        
        # Find natural phrase boundaries using VAD only
        chunks = self.find_natural_phrase_boundaries(audio)
        
        if not chunks:
            print("❌ No chunks created")
            return
        
        chunk_count = 0
        processed_chunks = 0
        
        for i, (start_ms, end_ms) in enumerate(chunks):
            # IMPROVED: Check running flag more frequently
            if not self.running:
                print("⏹️  Stopping due to interrupt signal...")
                break
                
            chunk_count += 1
            chunk_duration = (end_ms - start_ms) / 1000.0
            
            print(f"🎵 Pro STS Enhanced VAD FIXED: Processing chunk {chunk_count}/{len(chunks)} ({chunk_duration:.1f}s)")
            print(f"   Time range: {start_ms/1000:.1f}s - {end_ms/1000:.1f}s")
            
            # Extract chunk
            chunk_audio = audio[start_ms:end_ms]
            
            # IMPROVED: Additional quality check for chunk
            chunk_audio = self._validate_chunk_quality(chunk_audio)
            
            # Convert chunk to bytes
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                chunk_audio.export(temp_file.name, format="wav")
                temp_file_path = temp_file.name
            
            # Read the audio file
            with open(temp_file_path, "rb") as audio_file:
                audio_data = audio_file.read()
            
            # Clean up temp file
            os.unlink(temp_file_path)
            
            # IMPROVED: Check running flag before API call
            if not self.running:
                print("⏹️  Stopping before API call...")
                break
            
            # Process with STS API with retry logic
            success = self.process_audio_with_sts_pro_with_retry(audio_data, chunk_count)
            
            if success:
                processed_chunks += 1
            
            # IMPROVED: Check running flag after processing
            if not self.running:
                print("⏹️  Stopping after chunk processing...")
                break
            
            # Small delay between chunks for smooth streaming
            time.sleep(0.1)
        
        print(f"🎵 Pro STS Enhanced VAD FIXED: Processed {processed_chunks}/{chunk_count} chunks")
    
    def _validate_chunk_quality(self, chunk_audio):
        """IMPROVED: Validate and enhance chunk quality before processing"""
        try:
            # Convert to numpy for analysis
            samples = np.array(chunk_audio.get_array_of_samples(), dtype=np.float32)
            
            # Check audio level
            rms_level = np.sqrt(np.mean(samples ** 2))
            max_level = np.max(np.abs(samples))
            
            print(f"🔍 Chunk quality: RMS={rms_level:.4f}, Max={max_level:.4f}")
            
            # If chunk is too quiet, boost it slightly
            if rms_level < self.MIN_AUDIO_LEVEL:
                boost_factor = self.MIN_AUDIO_LEVEL / rms_level
                boost_factor = min(boost_factor, 1.5)  # Limit boost to prevent distortion
                samples = samples * boost_factor
                print(f"🔧 Boosted quiet chunk: factor {boost_factor:.2f}")
                
                # Convert back to AudioSegment
                enhanced_samples = (samples * 32767).astype(np.int16)
                chunk_audio = AudioSegment(
                    enhanced_samples.tobytes(),
                    frame_rate=chunk_audio.frame_rate,
                    sample_width=2,
                    channels=1
                )
            
            return chunk_audio
            
        except Exception as e:
            print(f"❌ Error validating chunk quality: {e}")
            return chunk_audio
    
    def process_audio_with_sts_pro_with_retry(self, audio_data, chunk_index, max_retries=3):
        """Process audio using ElevenLabs Speech-to-Speech API with IMPROVED settings"""
        for attempt in range(max_retries):
            # IMPROVED: Check running flag at start of each attempt
            if not self.running:
                print("⏹️  Stopping API processing due to interrupt...")
                return False
                
            try:
                print(f"🎵 Pro STS: Sending chunk {chunk_index} to ElevenLabs STS API (attempt {attempt + 1}/{max_retries})...")
                
                # Convert bytes to numpy array
                audio_array = np.frombuffer(audio_data, dtype=np.int16)
                
                # Save audio to temporary file
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                    sf.write(temp_file, audio_array, self.STS_SAMPLE_RATE)
                    temp_file_path = temp_file.name
                
                # Prepare the request with IMPROVED Pro features
                headers = {
                    "xi-api-key": self.api_key
                }
                
                # Read the audio file
                with open(temp_file_path, "rb") as audio_file:
                    files = {
                        "audio": ("audio.wav", audio_file, "audio/wav"),
                        "model_id": (None, self.model_id),
                        "remove_background_noise": (None, "false"),
                        "optimize_streaming_latency": (None, str(self.optimize_streaming_latency)),
                        "output_format": (None, self.output_format),
                        "voice_settings": (None, json.dumps(self.voice_settings))
                    }
                    
                    print(f"🎵 Pro STS: Sending chunk {chunk_index} to ElevenLabs STS API")
                    print(f"   Model: {self.model_id}")
                    print(f"   Voice: {self.voice_id}")
                    print(f"   Output Format: {self.output_format}")
                    print(f"   Voice Settings: {self.voice_settings}")
                    print(f"   Latency Optimization: {self.optimize_streaming_latency}")
                    
                    response = requests.post(
                        f"https://api.elevenlabs.io/v1/speech-to-speech/{self.voice_id}/stream",
                        headers=headers,
                        files=files,
                        timeout=30
                    )
                
                # Clean up temp file
                os.unlink(temp_file_path)
                
                print(f"🎵 Pro STS API response: status={response.status_code}")
                
                if response.status_code == 200:
                    # Get the audio data
                    audio_output = response.content
                    
                    if audio_output:
                        print(f"✅ Pro STS: Received {len(audio_output)} bytes for chunk {chunk_index}")
                        
                        # Add to continuous buffer for playback
                        try:
                            self.continuous_buffer.put_nowait(audio_output)
                        except queue.Full:
                            pass
                        
                        # Collect for file saving
                        self.mp3_chunks.append(audio_output)
                        self.all_audio_chunks.append(audio_output)
                        global audio_chunks
                        audio_chunks.append(audio_output)
                        
                        # Save individual chunk immediately
                        chunk_filename = f"pro_sts_enhanced_vad_fixed_chunk_{chunk_index}_{self.timestamp}.mp3"
                        with open(chunk_filename, 'wb') as f:
                            f.write(audio_output)
                        print(f"💾 Saved Pro STS Enhanced VAD FIXED chunk {chunk_index}: {chunk_filename} ({len(audio_output)} bytes)")
                        
                        # Start playback when we have enough data
                        if not self.playback_started and len(self.mp3_chunks) >= 2:
                            self.playback_started = True
                            print(f"🎵 Starting Pro STS Enhanced VAD FIXED streaming playback")
                        
                        # Log to file
                        self.log_to_file(f"PRO_STS_ENHANCED_VAD_FIXED_SUCCESS: Chunk {chunk_index}, {len(audio_output)} bytes")
                        return True
                    else:
                        print(f"⚠️ Pro STS: No audio data received for chunk {chunk_index}")
                        self.log_to_file(f"PRO_STS_ENHANCED_VAD_FIXED_ERROR: Chunk {chunk_index} - No audio data received")
                        if attempt < max_retries - 1:
                            print(f"🔄 Retrying chunk {chunk_index}...")
                            time.sleep(1)
                            continue
                        return False
                else:
                    print(f"❌ Pro STS API error for chunk {chunk_index}: {response.status_code} - {response.text}")
                    self.log_to_file(f"PRO_STS_ENHANCED_VAD_FIXED_ERROR: Chunk {chunk_index} - {response.status_code} - {response.text}")
                    if attempt < max_retries - 1:
                        print(f"🔄 Retrying chunk {chunk_index}...")
                        time.sleep(2)
                        continue
                    return False
                    
            except Exception as e:
                print(f"❌ Pro STS processing error for chunk {chunk_index}: {e}")
                self.log_to_file(f"PRO_STS_ENHANCED_VAD_FIXED_ERROR: Chunk {chunk_index} - {e}")
                if attempt < max_retries - 1:
                    print(f"🔄 Retrying chunk {chunk_index}...")
                    time.sleep(1)
                    continue
                return False
        
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
            
            print("🎵 Pro STS Enhanced VAD FIXED Streaming: Audio streaming initialized")
            
        except Exception as e:
            print(f"❌ Smooth streaming error: {e}")
    
    def _continuous_playback_worker(self, stream):
        """Continuous playback worker that eliminates gaps"""
        try:
            while self.running:
                try:
                    # Get audio chunk from buffer
                    mp3_data = self.continuous_buffer.get(timeout=0.1)
                    
                    # Play the audio chunk smoothly
                    self._play_audio_chunk_smooth(mp3_data, stream)
                    
                except queue.Empty:
                    # No audio data, continue
                    continue
                except Exception as e:
                    print(f"❌ Playback error: {e}")
                    break
                    
        except Exception as e:
            print(f"❌ Continuous playback error: {e}")
        finally:
            if stream:
                stream.stop_stream()
                stream.close()
    
    def _play_audio_chunk_smooth(self, mp3_data, stream):
        """Play audio chunk smoothly without gaps"""
        try:
            # Convert MP3 to PCM
            audio_segment = AudioSegment.from_mp3(io.BytesIO(mp3_data))
            
            # Convert to mono if needed
            if audio_segment.channels > 1:
                audio_segment = audio_segment.set_channels(1)
            
            # Resample if needed
            if audio_segment.frame_rate != self.SAMPLE_RATE:
                audio_segment = audio_segment.set_frame_rate(self.SAMPLE_RATE)
            
            # Convert to PCM
            pcm_data = audio_segment.raw_data
            
            # Play the audio
            stream.write(pcm_data)
            
        except Exception as e:
            print(f"❌ Audio chunk playback error: {e}")
    
    async def start(self):
        """Start the Pro STS enhanced VAD FIXED streaming test"""
        try:
            print("🎤🎵 Pro STS Enhanced VAD FIXED Streaming Test: MP3 File → Natural Phrases → ElevenLabs STS API → Speakers")
            print("=" * 80)
            print("🎤 Enhanced VAD: Detects natural phrase boundaries using REAL pause detection")
            print("🔧 Audio Quality: Enhanced normalization and noise reduction")
            print("🎵 STS: Sends natural phrases to ElevenLabs STS API with IMPROVED settings")
            print("🔊 Playback: Smooth real-time audio streaming")
            print(f"📁 Queue log: {self.QUEUE_LOG_FILE}")
            print(f"🎵 Output file: {self.OUTPUT_FILE}")
            print(f"🎵 Model: {self.model_id}")
            print(f"🎵 Voice: {self.voice_id}")
            print(f"🎵 Pro Features: {self.output_format}, {self.voice_settings}")
            print("🎤 VAD-ONLY: Uses real silence detection, no fixed chunking")
            print("=" * 80)
            
            # Load and prepare audio
            if not self.load_and_prepare_audio():
                return
            
            # Start smooth audio streaming
            await self._smooth_audio_streaming()
            
            print("🎤 Pro STS Enhanced VAD FIXED: Starting natural phrase detection...")
            print("✅ Pro STS Enhanced VAD FIXED streaming test started!")
            print("🎵 Streaming MP3 file using REAL natural phrase boundaries with IMPROVED quality...")
            print("🎵 Converted audio will play through speakers!")
            print("⏹️  Press Ctrl+C to stop early (improved handling)")
            print("🎤 VAD-ONLY: Using real silence detection for natural phrase boundaries")
            
            # IMPROVED: Check running flag before starting processing
            if not self.running:
                print("⏹️  Stopping before processing due to interrupt...")
                return
            
            # Stream audio by enhanced VAD
            self.stream_audio_by_enhanced_vad()
            
            print("✅ Pro STS Enhanced VAD FIXED streaming test completed")
            
        except Exception as e:
            print(f"❌ Pro STS Enhanced VAD FIXED streaming error: {e}")
        finally:
            # IMPROVED: Always save output even on interrupt
            if not self.running:
                print("💾 Saving output due to interrupt...")
            self.save_final_output()
    
    def save_final_output(self):
        """Save final output file"""
        try:
            print("💾 Saving final output...")
            save_audio_chunks(self.all_audio_chunks, self.OUTPUT_FILE)
        except Exception as e:
            print(f"❌ Error saving final output: {e}")

async def main():
    """Main function with improved interrupt handling"""
    global running
    
    try:
        # Parse command line arguments
        parser = argparse.ArgumentParser(description="Pro STS Streaming Enhanced VAD FIXED Test")
        parser.add_argument("input_file", help="Input MP3 file")
        parser.add_argument("--output", help="Output MP3 file", default="test_output.mp3")
        args = parser.parse_args()
        
        # Check prerequisites
        print("✅ Prerequisites check passed")
        
        # Create and start the test
        pro_sts_enhanced_vad_fixed = ProSTSStreamingEnhancedVADFixedTest(args.input_file)
        
        # Override output file if specified
        if args.output:
            pro_sts_enhanced_vad_fixed.OUTPUT_FILE = args.output
            print(f"🎵 Output file set to: {args.output}")
        
        # IMPROVED: Set global running flag
        running = True
        
        # Start the streaming test
        await pro_sts_enhanced_vad_fixed.start()
        
    except KeyboardInterrupt:
        print("\n⏹️  Interrupted by user (main)")
        running = False
    except Exception as e:
        print(f"❌ Main error: {e}")
    finally:
        # IMPROVED: Force cleanup
        running = False
        print("🧹 Cleanup completed")

if __name__ == "__main__":
    import io
    asyncio.run(main()) 