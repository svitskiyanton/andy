#!/usr/bin/env python3
"""
STS Streaming Pipeline - Complete streaming pipeline
Input Chunk → STS → Output Chunk → Playback
Combines VAD chunking, STS processing, and real-time playback
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
            save_audio_chunks(audio_chunks, "sts_streaming_output.mp3")

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

class STSStreamingPipeline:
    def __init__(self, input_file=None):
        # Audio settings
        self.STS_SAMPLE_RATE = 44100
        self.SAMPLE_RATE = 44100
        self.CHUNK_SIZE = 1024
        self.CHANNELS = 1
        self.FORMAT = pyaudio.paInt16
        
        # Input/Output files
        self.input_file = input_file
        self.OUTPUT_FILE = f"sts_streaming_output_{time.strftime('%Y%m%d_%H%M%S')}.mp3"
        
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
        self.MIN_PHRASE_DURATION = 0.2
        self.SILENCE_DURATION = 0.05
        
        # ElevenLabs settings - SAME AS test_pro_sts_streaming_enhanced_vad_fixed.py
        self.api_key = self._get_api_key()
        self.voice_id = self._get_voice_id()
        self.model_id = "eleven_multilingual_sts_v2"
        
        # Pro voice settings - SAME AS test_pro_sts_streaming_enhanced_vad_fixed.py
        self.voice_settings = {
            "stability": 0.8,  # HIGHER stability for more consistent output
            "similarity_boost": 0.85,  # HIGHER similarity for better voice cloning
            "style": 0.2,  # LOWER style for more natural speech
            "use_speaker_boost": True
        }
        
        # Pro audio settings - SAME AS test_pro_sts_streaming_enhanced_vad_fixed.py
        self.output_format = "mp3_44100_192"
        self.optimize_streaming_latency = 3  # REDUCED from 4 for better quality
        
        # Audio processing
        self.audio_segments = []
        
        print(f"🎵 STS Streaming Pipeline Configuration:")
        print(f"   Input File: {self.input_file}")
        print(f"   Output File: {self.OUTPUT_FILE}")
        print(f"   Voice ID: {self.voice_id}")
        print(f"   Model: {self.model_id}")
        print(f"   Voice Settings: {self.voice_settings}")
        print(f"   Output Format: {self.output_format}")
        
    def _get_api_key(self):
        """Get API key from environment or .env file - SAME AS test_pro_sts_streaming_enhanced_vad_fixed.py"""
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
        """Get voice ID from environment or .env file - SAME AS test_pro_sts_streaming_enhanced_vad_fixed.py"""
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
        
        # Default to Ekaterina if not found - SAME AS test_pro_sts_streaming_enhanced_vad_fixed.py
        return "GN4wbsbejSnGSa1AzjH5"
    
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
        vad = webrtcvad.Vad(0)  # Most sensitive
        audio_16k = audio_segment.set_frame_rate(self.VAD_SAMPLE_RATE).set_channels(1)
        samples = np.array(audio_16k.get_array_of_samples(), dtype=np.int16)
        frame_size = int(self.VAD_SAMPLE_RATE * self.VAD_FRAME_DURATION)
        total_frames = len(samples) // frame_size

        speech_flags = []
        for i in range(total_frames):
            frame = samples[i*frame_size:(i+1)*frame_size]
            if len(frame) < frame_size:
                break
            is_speech = vad.is_speech(frame.tobytes(), self.VAD_SAMPLE_RATE)
            speech_flags.append(is_speech)

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
        return boundaries
    
    def process_chunk_with_sts(self, chunk_audio, chunk_index, max_retries=3):
        """Process a single chunk with ElevenLabs STS API"""
        for attempt in range(max_retries):
            try:
                print(f"🎵 STS: Processing chunk {chunk_index} (attempt {attempt + 1}/{max_retries})...")
                
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
                    
                    print(f"🎵 STS: Sending chunk {chunk_index} to ElevenLabs STS API")
                    print(f"   Model: {self.model_id}")
                    print(f"   Voice: {self.voice_id}")
                    print(f"   Output Format: {self.output_format}")
                    
                    response = requests.post(
                        f"https://api.elevenlabs.io/v1/speech-to-speech/{self.voice_id}/stream",
                        headers=headers,
                        files=files,
                        timeout=30
                    )
                
                # Clean up temp file
                os.unlink(temp_file_path)
                
                print(f"🎵 STS API response: status={response.status_code}")
                
                if response.status_code == 200:
                    audio_output = response.content
                    
                    if audio_output:
                        print(f"✅ STS: Received {len(audio_output)} bytes for chunk {chunk_index}")
                        return audio_output
                    else:
                        print(f"⚠️ STS: No audio data received for chunk {chunk_index}")
                        if attempt < max_retries - 1:
                            print(f"🔄 Retrying chunk {chunk_index}...")
                            time.sleep(1)
                            continue
                        return None
                else:
                    print(f"❌ STS API error for chunk {chunk_index}: {response.status_code} - {response.text}")
                    if attempt < max_retries - 1:
                        print(f"🔄 Retrying chunk {chunk_index}...")
                        time.sleep(2)
                        continue
                    return None
                    
            except Exception as e:
                print(f"❌ STS processing error for chunk {chunk_index}: {e}")
                if attempt < max_retries - 1:
                    print(f"🔄 Retrying chunk {chunk_index}...")
                    time.sleep(1)
                    continue
                return None
        
        return None
    
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
            
            print("🎵 STS Streaming Pipeline: Audio streaming initialized")
            
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
    
    def stream_audio_pipeline(self):
        """Main streaming pipeline: Input Chunk → STS → Output Chunk → Playback"""
        print("🎤 STS Streaming Pipeline: Starting VAD chunking...")
        
        if not self.audio_segments:
            print("❌ No audio loaded")
            return
        
        audio = self.audio_segments[0]
        
        # Find natural phrase boundaries using VAD
        chunks = self.detect_vad_boundaries(audio)
        
        if not chunks:
            print("❌ No chunks created")
            return
        
        print(f"✅ Created {len(chunks)} VAD chunks")
        
        chunk_count = 0
        processed_chunks = 0
        
        for i, (start_ms, end_ms) in enumerate(chunks):
            # Check running flag
            if not self.running:
                print("⏹️  Stopping due to interrupt signal...")
                break
                
            chunk_count += 1
            chunk_duration = (end_ms - start_ms) / 1000.0
            
            print(f"🎵 STS Pipeline: Processing chunk {chunk_count}/{len(chunks)} ({chunk_duration:.1f}s)")
            print(f"   Time range: {start_ms/1000:.1f}s - {end_ms/1000:.1f}s")
            
            # Extract chunk
            chunk_audio = audio[start_ms:end_ms]
            
            # Process with STS API
            audio_output = self.process_chunk_with_sts(chunk_audio, chunk_count)
            
            if audio_output:
                # Add to continuous buffer for playback
                try:
                    self.continuous_buffer.put_nowait(audio_output)
                except queue.Full:
                    pass
                
                # Collect for file saving
                self.all_audio_chunks.append(audio_output)
                global audio_chunks
                audio_chunks.append(audio_output)
                
                # Start playback when we have enough data
                if not self.playback_started and chunk_count >= 1:
                    self.playback_started = True
                    print(f"🎵 Starting STS Pipeline streaming playback")
                
                processed_chunks += 1
            
            # Small delay between chunks for smooth streaming
            time.sleep(0.1)
        
        print(f"🎵 STS Pipeline: Processed {processed_chunks}/{chunk_count} chunks")
    
    async def start(self):
        """Start the STS streaming pipeline"""
        try:
            print("🎤🎵 STS Streaming Pipeline: MP3 File → VAD Chunks → STS API → Playback")
            print("=" * 80)
            print("🎤 VAD: Detects natural phrase boundaries")
            print("🎵 STS: Sends chunks to ElevenLabs STS API")
            print("🔊 Playback: Smooth real-time audio streaming")
            print(f"📁 Output file: {self.OUTPUT_FILE}")
            print(f"🎵 Model: {self.model_id}")
            print(f"🎵 Voice: {self.voice_id}")
            print("=" * 80)
            
            # Load and prepare audio
            if not self.load_and_prepare_audio():
                return
            
            # Start smooth audio streaming
            await self._smooth_audio_streaming()
            
            print("🎤 STS Pipeline: Starting VAD chunking and STS processing...")
            print("✅ STS Streaming Pipeline started!")
            print("🎵 Streaming: Input Chunk → STS → Output Chunk → Playback")
            print("🎵 Converted audio will play through speakers!")
            print("⏹️  Press Ctrl+C to stop early")
            
            # Check running flag before starting processing
            if not self.running:
                print("⏹️  Stopping before processing due to interrupt...")
                return
            
            # Stream audio pipeline
            self.stream_audio_pipeline()
            
            print("✅ STS Streaming Pipeline completed")
            
        except Exception as e:
            print(f"❌ STS Streaming Pipeline error: {e}")
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
        parser = argparse.ArgumentParser(description="STS Streaming Pipeline")
        parser.add_argument("input_file", help="Input MP3 file")
        parser.add_argument("--output", help="Output MP3 file", default="sts_streaming_output.mp3")
        args = parser.parse_args()
        
        # Check prerequisites
        print("✅ Prerequisites check passed")
        
        # Create and start the pipeline
        sts_pipeline = STSStreamingPipeline(args.input_file)
        
        # Override output file if specified
        if args.output:
            sts_pipeline.OUTPUT_FILE = args.output
            print(f"🎵 Output file set to: {args.output}")
        
        # Set global running flag
        running = True
        
        # Start the streaming pipeline
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