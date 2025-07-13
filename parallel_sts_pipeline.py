#!/usr/bin/env python3
"""
Parallel STS Pipeline - Concurrent processing with order preservation
Uses ElevenLabs PRO subscription (10 concurrent requests)
Maintains proper chunk order for playback
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

# Global variables for cleanup
running = True
audio_chunks = []

def signal_handler(signum, frame):
    """Handle interrupt signal"""
    global running
    print("\n⏹️  Received interrupt signal, exiting...")
    running = False

def cleanup_handler():
    """Cleanup handler for atexit"""
    global running, audio_chunks
    if running:
        running = False
        print("💾 Saving output on exit...")
        if audio_chunks:
            save_audio_chunks(audio_chunks, "parallel_sts_output.mp3")

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

class ParallelSTSPipeline:
    def __init__(self, input_file=None):
        # Audio settings
        self.STS_SAMPLE_RATE = 44100
        self.SAMPLE_RATE = 44100
        self.CHUNK_SIZE = 1024
        self.CHANNELS = 1
        self.FORMAT = pyaudio.paInt16
        
        # Input/Output files
        self.input_file = input_file
        self.OUTPUT_FILE = f"parallel_sts_output_{time.strftime('%Y%m%d_%H%M%S')}.mp3"
        
        # Audio processing
        self.audio = pyaudio.PyAudio()
        self.continuous_buffer = queue.Queue(maxsize=50)
        self.playback_thread = None
        self.running = True
        self.playback_started = False
        self.all_audio_chunks = []
        
        # VAD settings
        self.VAD_SAMPLE_RATE = 16000
        self.VAD_FRAME_DURATION = 0.03
        self.VAD_FRAME_SIZE = int(self.VAD_SAMPLE_RATE * self.VAD_FRAME_DURATION)
        self.MIN_PHRASE_DURATION = 0.5
        self.SILENCE_DURATION = 0.1
        
        # ElevenLabs PRO settings
        self.api_key = self._get_api_key()
        self.voice_id = "21m00Tcm4TlvDq8ikWAM"
        self.model_id = "eleven_multilingual_sts_v2"
        
        # Rachel voice settings updated per user request
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
        self.MAX_CONCURRENT_REQUESTS = 5  # Reduced from 10 to 5
        self.request_semaphore = asyncio.Semaphore(self.MAX_CONCURRENT_REQUESTS)
        self.order_queue = queue.Queue()  # Maintain chunk order
        self.completed_chunks: Dict[int, bytes] = {}  # Store completed chunks by index
        self.next_chunk_index = 0  # Track next chunk to play
        self.processing_complete = False  # Track when all processing is done
        
        # Performance tracking
        self.active_requests = 0
        self.completed_requests = 0
        self.failed_requests = 0
        
        print(f"🎵 Parallel STS Pipeline Configuration:")
        print(f"   Input File: {self.input_file}")
        print(f"   Output File: {self.OUTPUT_FILE}")
        print(f"   Voice ID: {self.voice_id}")
        print(f"   Model: {self.model_id}")
        print(f"   Max Concurrent Requests: {self.MAX_CONCURRENT_REQUESTS}")
        print(f"   Voice Settings: {self.voice_settings}")
        print(f"   Output Format: {self.output_format}")
        
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
        return "21m00Tcm4TlvDq8ikWAM"
    
    def load_and_prepare_audio(self):
        """Load MP3 file and prepare it for processing"""
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
            
            self.audio_segments = [audio]
            print("✅ Audio prepared for VAD processing")
            return True
            
        except Exception as e:
            print(f"❌ Error loading audio: {e}")
            return False
    
    def detect_vad_boundaries(self, audio_segment):
        """Detect phrase boundaries using VAD and return list of (start_ms, end_ms) tuples"""
        print("🔍 Starting VAD boundary detection...")
        
        vad = webrtcvad.Vad(2)  # Medium sensitivity
        audio_16k = audio_segment.set_frame_rate(self.VAD_SAMPLE_RATE).set_channels(1)
        samples = np.array(audio_16k.get_array_of_samples(), dtype=np.int16)
        frame_size = int(self.VAD_SAMPLE_RATE * self.VAD_FRAME_DURATION)
        total_frames = len(samples) // frame_size

        print(f"🔍 Total frames: {total_frames}, Frame size: {frame_size}")

        speech_flags = []
        for i in range(total_frames):
            frame = samples[i*frame_size:(i+1)*frame_size]
            if len(frame) < frame_size:
                break
            is_speech = vad.is_speech(frame.tobytes(), self.VAD_SAMPLE_RATE)
            speech_flags.append(is_speech)

        print(f"🔍 Speech flags: {len(speech_flags)} frames")

        # Find boundaries: split at runs of silence
        boundaries = []
        in_speech = False
        chunk_start = 0
        for i, is_speech in enumerate(speech_flags):
            t = i * self.VAD_FRAME_DURATION
            if is_speech and not in_speech:
                # Start of speech
                chunk_start = t
                in_speech = True
            elif not is_speech and in_speech:
                # End of speech (pause)
                chunk_end = t
                if chunk_end - chunk_start >= self.MIN_PHRASE_DURATION:
                    boundaries.append((int(chunk_start*1000), int(chunk_end*1000)))
                in_speech = False
        # Handle last chunk
        if in_speech:
            chunk_end = total_frames * self.VAD_FRAME_DURATION
            if chunk_end - chunk_start >= self.MIN_PHRASE_DURATION:
                boundaries.append((int(chunk_start*1000), int(chunk_end*1000)))
        
        print(f"🔍 Found {len(boundaries)} VAD boundaries")
        for i, (start, end) in enumerate(boundaries):
            duration = (end - start) / 1000.0
            print(f"   Boundary {i+1}: {start/1000:.2f}s - {end/1000:.2f}s ({duration:.2f}s)")
        
        return boundaries
    
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
    
    def save_chunk_to_file(self, chunk_audio_data, chunk_index):
        """Save chunk to file for debugging"""
        try:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"parallel_chunk_{chunk_index:03d}_{timestamp}.wav"
            filepath = os.path.join("voice_emulation", filename)
            
            # Convert raw PCM data back to AudioSegment
            audio_segment = AudioSegment(
                data=chunk_audio_data,
                sample_width=2,  # 16-bit
                frame_rate=self.SAMPLE_RATE,
                channels=1
            )
            
            # Save as WAV file
            audio_segment.export(filepath, format="wav")
            print(f"💾 Saved parallel chunk {chunk_index}: {filename}")
            return filepath
            
        except Exception as e:
            print(f"❌ Error saving chunk {chunk_index}: {e}")
            return None
    
    async def process_chunks_parallel(self, chunks):
        """Process all chunks in parallel with real-time streaming"""
        print(f"🚀 Starting parallel STS processing for {len(chunks)} chunks...")
        
        # Create tasks for all chunks
        tasks = []
        for i, (start_ms, end_ms) in enumerate(chunks):
            chunk_audio = self.audio_segments[0][start_ms:end_ms]
            task = asyncio.create_task(self.process_chunk_with_sts_async(chunk_audio, i+1))
            tasks.append((i+1, task))  # Store chunk index with task
        
        # Start playback immediately when first chunk is ready
        self.playback_started = True
        print(f"🎵 Starting real-time streaming playback")
        
        # Process tasks as they complete and stream immediately
        completed_count = 0
        for chunk_index, task in tasks:
            try:
                result = await task
                if result:
                    # Stream chunk immediately when ready
                    await self.stream_chunk_immediately(chunk_index, result)
                    completed_count += 1
                    print(f"✅ Chunk {chunk_index} completed and streamed immediately")
                else:
                    print(f"❌ Chunk {chunk_index} failed")
            except Exception as e:
                print(f"❌ Task for chunk {chunk_index} failed: {e}")
        
        print(f"📊 Parallel processing completed:")
        print(f"   Total chunks: {len(chunks)}")
        print(f"   Completed: {completed_count}")
        print(f"   Failed: {len(chunks) - completed_count}")
        
        # Wait for all chunks to be played
        print("⏳ Waiting for all chunks to finish playing...")
        while not self.continuous_buffer.empty():
            await asyncio.sleep(0.1)
        
        # Give extra time for final chunks to play
        await asyncio.sleep(2.0)
        self.processing_complete = True
        print("✅ All chunks have been played")
    
    async def stream_chunk_immediately(self, chunk_index, audio_output):
        """Stream chunk immediately when it's ready"""
        try:
            # Add to playback buffer immediately
            self.continuous_buffer.put_nowait(audio_output)
            print(f"🔍 Chunk {chunk_index} added to playback buffer")
            
            # Add 0.5s silence after the chunk for spacing
            silence_duration = int(0.5 * self.SAMPLE_RATE * 2)  # 0.5s of silence (16-bit)
            silence_data = b'\x00' * silence_duration
            self.continuous_buffer.put_nowait(silence_data)
            print(f"🔇 Added 0.5s silence after chunk {chunk_index}")
            
            # Collect for file saving
            self.all_audio_chunks.append(audio_output)
            global audio_chunks
            audio_chunks.append(audio_output)
            
        except queue.Full:
            print(f"⚠️ Playback buffer full, skipping chunk {chunk_index}")
        except Exception as e:
            print(f"❌ Error streaming chunk {chunk_index}: {e}")
    
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
            
            print("🎵 Parallel STS Pipeline: Audio streaming initialized")
            
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
                    # No audio data, but check if processing is complete
                    if self.processing_complete and self.continuous_buffer.empty():
                        print("🎵 Playback complete - all chunks processed and played")
                        break
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
    
    def _play_audio_chunk_smooth(self, audio_data, stream):
        """Play audio chunk or silence smoothly"""
        try:
            # Check if this is silence data (all zeros)
            if all(b == 0 for b in audio_data[:100]):  # Check first 100 bytes
                # This is silence data
                silence_duration = len(audio_data) / (self.SAMPLE_RATE * 2)  # 16-bit
                print(f"🔇 Playing silence ({silence_duration:.1f}s)")
                stream.write(audio_data)
                print(f"✅ Silence played successfully")
            else:
                # This is MP3 audio data
                print(f"🔊 Playing audio chunk ({len(audio_data)} bytes)")
                
                # Convert MP3 to PCM
                audio_segment = AudioSegment.from_mp3(io.BytesIO(audio_data))
                
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
                print(f"✅ Audio chunk played successfully")
            
        except Exception as e:
            print(f"❌ Audio playback error: {e}")
    
    async def stream_audio_pipeline_parallel(self):
        """Main parallel streaming pipeline"""
        print("🎤 Parallel STS Pipeline: Starting VAD chunking...")
        
        if not self.audio_segments:
            print("❌ No audio loaded")
            return
        
        audio = self.audio_segments[0]
        
        # Find natural phrase boundaries using VAD
        print("🔍 Detecting VAD boundaries...")
        chunks = self.detect_vad_boundaries(audio)
        
        if not chunks:
            print("❌ No chunks created")
            return
        
        print(f"✅ Created {len(chunks)} VAD chunks")
        
        # Process chunks in parallel with real-time streaming
        await self.process_chunks_parallel(chunks)
        
        print(f"🎵 Parallel STS Pipeline: All chunks processed")
    
    async def start(self):
        """Start the parallel STS streaming pipeline"""
        try:
            print("🎤🎵 Parallel STS Pipeline: MP3 File → VAD Chunks → Parallel STS API → Playback")
            print("=" * 80)
            print("🎤 VAD: Detects natural phrase boundaries")
            print("🚀 Parallel STS: Sends chunks concurrently (max 10)")
            print("🔊 Playback: Smooth real-time audio streaming")
            print(f"📁 Output file: {self.OUTPUT_FILE}")
            print(f"🎵 Model: {self.model_id}")
            print(f"🎵 Voice: {self.voice_id}")
            print(f"⚡ Max Concurrent Requests: {self.MAX_CONCURRENT_REQUESTS}")
            print("=" * 80)
            
            # Load and prepare audio
            if not self.load_and_prepare_audio():
                return
            
            # Start smooth audio streaming
            await self._smooth_audio_streaming()
            
            print("🎤 Parallel STS Pipeline: Starting VAD chunking and parallel STS processing...")
            print("✅ Parallel STS Pipeline started!")
            print("🚀 Processing: Input Chunks → Parallel STS → Output Chunks → Playback")
            print("🎵 Converted audio will play through speakers!")
            print("⏹️  Press Ctrl+C to stop early")
            
            # Check running flag before starting processing
            if not self.running:
                print("⏹️  Stopping before processing due to interrupt...")
                return
            
            # Stream audio pipeline in parallel
            await self.stream_audio_pipeline_parallel()
            
            print("✅ Parallel STS Pipeline completed")
            
        except Exception as e:
            print(f"❌ Parallel STS Pipeline error: {e}")
        finally:
            # Always save output even on interrupt
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
        parser = argparse.ArgumentParser(description="Parallel STS Pipeline")
        parser.add_argument("input_file", help="Input MP3 file")
        parser.add_argument("--output", help="Output MP3 file", default="parallel_sts_output.mp3")
        args = parser.parse_args()
        
        # Check prerequisites
        print("✅ Prerequisites check passed")
        
        # Create and start the pipeline
        sts_pipeline = ParallelSTSPipeline(args.input_file)
        
        # Override output file if specified
        if args.output:
            sts_pipeline.OUTPUT_FILE = args.output
            print(f"🎵 Output file set to: {args.output}")
        
        # Set global running flag
        running = True
        
        # Start the parallel streaming pipeline
        await sts_pipeline.start()
        
    except KeyboardInterrupt:
        print("\n⏹️  Interrupted by user (main)")
        running = False
    except Exception as e:
        print(f"❌ Main error: {e}")
    finally:
        # Force cleanup
        running = False
        print("🧹 Cleanup completed")

if __name__ == "__main__":
    import io
    asyncio.run(main()) 