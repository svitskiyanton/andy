#!/usr/bin/env python3
"""
Test Script: Batch STS with ElevenLabs Pro Features
Takes MP3 file as input → ElevenLabs STS API → Output MP3 file
Uses Pro subscription capabilities: 192kbps MP3, advanced voice settings, optimized processing
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
import argparse
import sys

# Load environment variables
load_dotenv()

class ProSTSBatchTest:
    def __init__(self, input_file=None):
        # Audio settings (Pro quality)
        self.FORMAT = pyaudio.paFloat32
        self.CHANNELS = 1
        self.SAMPLE_RATE = 44100  # Pro supports 44.1kHz
        self.CHUNK_SIZE = 2048  # Larger chunk for Pro
        
        # Input/Output files
        self.input_file = input_file
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.output_file = f"test_pro_sts_batch_output_{timestamp}.mp3"
        
        # File logging
        self.QUEUE_LOG_FILE = "test_pro_sts_batch_log.txt"
        
        # Control
        self.running = True
        
        # STS settings (Pro optimized)
        self.STS_SAMPLE_RATE = 44100  # Pro quality
        self.STS_CHANNELS = 1
        self.STS_FORMAT = pyaudio.paInt16
        self.STS_CHUNK_SIZE = 4096  # Larger for Pro
        
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
        self.processed_chunks = []
        
        print(f"🎵 Pro STS Batch Test Configuration:")
        print(f"   Input File: {self.input_file}")
        print(f"   Output File: {self.output_file}")
        print(f"   Voice ID: {self.voice_id}")
        print(f"   Model: {self.model_id}")
        print(f"   Output Format: {self.output_format}")
        print(f"   Sample Rate: {self.STS_SAMPLE_RATE}Hz")
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
        """Load MP3 file and prepare it for STS processing"""
        try:
            print(f"📁 Loading audio file: {self.input_file}")
            
            # Load the MP3 file
            audio = AudioSegment.from_mp3(self.input_file)
            print(f"✅ Loaded audio: {len(audio)}ms ({len(audio)/1000:.1f}s)")
            
            # Convert to mono if stereo
            if audio.channels > 1:
                audio = audio.set_channels(1)
                print("🔄 Converted to mono")
            
            # Resample to 44.1kHz if needed
            if audio.frame_rate != self.STS_SAMPLE_RATE:
                audio = audio.set_frame_rate(self.STS_SAMPLE_RATE)
                print(f"🔄 Resampled to {self.STS_SAMPLE_RATE}Hz")
            
            # Split into chunks (max 5 minutes per chunk for stability)
            max_chunk_duration = 5 * 60 * 1000  # 5 minutes in milliseconds
            chunks = []
            
            if len(audio) <= max_chunk_duration:
                # Single chunk
                chunks = [audio]
                print("📦 Single chunk processing")
            else:
                # Multiple chunks
                chunk_count = (len(audio) + max_chunk_duration - 1) // max_chunk_duration
                for i in range(chunk_count):
                    start_time = i * max_chunk_duration
                    end_time = min((i + 1) * max_chunk_duration, len(audio))
                    chunk = audio[start_time:end_time]
                    chunks.append(chunk)
                print(f"📦 Split into {len(chunks)} chunks")
            
            self.audio_segments = chunks
            print(f"✅ Audio prepared: {len(chunks)} chunks ready for processing")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading audio file: {e}")
            return False
    
    def process_audio_chunk_with_sts_pro(self, audio_segment, chunk_index):
        """Process audio chunk using ElevenLabs Speech-to-Speech API with Pro features"""
        try:
            print(f"🎵 Pro STS: Processing chunk {chunk_index + 1}/{len(self.audio_segments)} ({len(audio_segment)}ms)")
            
            # Export audio segment to WAV format
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                audio_segment.export(temp_file.name, format="wav")
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
                
                print(f"🎵 Pro STS: Sending chunk {chunk_index + 1} to ElevenLabs STS API")
                print(f"   Model: {self.model_id}")
                print(f"   Voice: {self.voice_id}")
                print(f"   Output Format: {self.output_format}")
                print(f"   Voice Settings: {self.voice_settings}")
                
                response = requests.post(
                    f"https://api.elevenlabs.io/v1/speech-to-speech/{self.voice_id}/stream",
                    headers=headers,
                    files=files
                )
            
            # Clean up temp file
            os.unlink(temp_file_path)
            
            print(f"🎵 Pro STS API response: status={response.status_code}")
            
            if response.status_code == 200:
                # Get the audio data
                audio_output = response.content
                
                if audio_output:
                    print(f"✅ Pro STS: Received {len(audio_output)} bytes for chunk {chunk_index + 1}")
                    
                    # Save individual chunk
                    chunk_filename = f"pro_sts_batch_chunk_{chunk_index + 1}_{self.timestamp}.mp3"
                    with open(chunk_filename, 'wb') as f:
                        f.write(audio_output)
                    print(f"💾 Saved chunk {chunk_index + 1}: {chunk_filename} ({len(audio_output)} bytes)")
                    
                    # Store for final combination
                    self.processed_chunks.append(audio_output)
                    
                    # Log to file
                    self.log_to_file(f"PRO_STS_BATCH_SUCCESS: Chunk {chunk_index + 1}, {len(audio_output)} bytes")
                    
                    return True
                else:
                    print(f"⚠️ Pro STS: No audio data received for chunk {chunk_index + 1}")
                    self.log_to_file(f"PRO_STS_BATCH_ERROR: Chunk {chunk_index + 1} - No audio data received")
                    return False
            else:
                print(f"❌ Pro STS API error for chunk {chunk_index + 1}: {response.status_code} - {response.text}")
                self.log_to_file(f"PRO_STS_BATCH_ERROR: Chunk {chunk_index + 1} - {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            print(f"❌ Pro STS processing error for chunk {chunk_index + 1}: {e}")
            self.log_to_file(f"PRO_STS_BATCH_ERROR: Chunk {chunk_index + 1} - {e}")
            return False
    
    def process_all_chunks(self):
        """Process all audio chunks sequentially"""
        print(f"🎵 Pro STS Batch: Starting processing of {len(self.audio_segments)} chunks...")
        
        successful_chunks = 0
        total_chunks = len(self.audio_segments)
        
        for i, audio_segment in enumerate(self.audio_segments):
            print(f"\n🔄 Processing chunk {i + 1}/{total_chunks}")
            
            # Process the chunk
            success = self.process_audio_chunk_with_sts_pro(audio_segment, i)
            
            if success:
                successful_chunks += 1
                print(f"✅ Chunk {i + 1}/{total_chunks} processed successfully")
            else:
                print(f"❌ Chunk {i + 1}/{total_chunks} failed")
            
            # Small delay between chunks to avoid rate limiting
            if i < total_chunks - 1:
                print("⏳ Waiting 1 second before next chunk...")
                time.sleep(1)
        
        print(f"\n🎵 Pro STS Batch: Processing complete!")
        print(f"   Total chunks: {total_chunks}")
        print(f"   Successful: {successful_chunks}")
        print(f"   Failed: {total_chunks - successful_chunks}")
        
        return successful_chunks > 0
    
    def combine_and_save_output(self):
        """Combine all processed chunks and save to output file"""
        try:
            if not self.processed_chunks:
                print("⚠️ No processed chunks to combine")
                return False
            
            print(f"💾 Combining {len(self.processed_chunks)} processed chunks...")
            
            # Combine all audio chunks
            combined_audio = b''.join(self.processed_chunks)
            
            # Save as MP3
            with open(self.output_file, 'wb') as f:
                f.write(combined_audio)
            
            print(f"✅ Combined audio saved to: {self.output_file}")
            print(f"📊 Total audio size: {len(combined_audio)} bytes")
            
            # Log final stats
            self.log_to_file(f"FINAL: Combined {len(self.processed_chunks)} chunks, {len(combined_audio)} bytes")
            
            return True
            
        except Exception as e:
            print(f"❌ Error combining output: {e}")
            self.log_to_file(f"ERROR: Failed to combine output - {e}")
            return False
    
    async def start(self):
        """Start the Pro STS batch test"""
        print("🎤🎵 Pro STS Batch Test: MP3 File → ElevenLabs STS API → Output File")
        print("=" * 70)
        print("🎤 STS: Loads MP3 file → splits into chunks → sends to ElevenLabs STS API")
        print("🎵 TTS: Receives converted audio → combines → saves to output file")
        print(f"📁 Queue log: {self.QUEUE_LOG_FILE}")
        print(f"🎵 Output file: {self.output_file}")
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
        
        # Process all chunks
        if not self.process_all_chunks():
            print("❌ Failed to process audio chunks")
            return
        
        # Combine and save output
        if not self.combine_and_save_output():
            print("❌ Failed to save output file")
            return
        
        print("✅ Pro STS batch test completed successfully!")
        print(f"🎵 Output saved to: {self.output_file}")

async def main():
    """Main function"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Pro STS Batch Test - Process MP3 file with ElevenLabs STS API')
    parser.add_argument('input_file', help='Input MP3 file path')
    parser.add_argument('--output', '-o', help='Output file path (optional)')
    
    args = parser.parse_args()
    
    # Check prerequisites
    if not os.getenv("ELEVENLABS_API_KEY"):
        print("❌ ELEVENLABS_API_KEY not found")
        print("   Please set your ElevenLabs API key in environment or .env file")
        return
    
    print("✅ Prerequisites check passed")
    
    # Create and start Pro STS batch test
    pro_sts_batch = ProSTSBatchTest(args.input_file)
    if args.output:
        pro_sts_batch.output_file = args.output
    
    await pro_sts_batch.start()

if __name__ == "__main__":
    asyncio.run(main()) 