#!/usr/bin/env python3
"""
STS Chunk Streamer - Stream VAD chunks to ElevenLabs STS API
Takes chunks from input directory, sends to STS API, saves output chunks
Uses same credentials and parameters as test_pro_sts_streaming_enhanced_vad_fixed.py
"""

import os
import sys
import time
import json
import argparse
import tempfile
import requests
from pydub import AudioSegment

class STSChunkStreamer:
    def __init__(self):
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
        
        # Audio settings - SAME AS test_pro_sts_streaming_enhanced_vad_fixed.py
        self.STS_SAMPLE_RATE = 44100
        
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
    
    def process_chunk_with_sts(self, chunk_file, chunk_index, max_retries=3):
        """Process a single chunk with ElevenLabs STS API"""
        for attempt in range(max_retries):
            try:
                print(f"🎵 STS: Processing chunk {chunk_index} (attempt {attempt + 1}/{max_retries})...")
                
                # Load the chunk
                audio_segment = AudioSegment.from_mp3(chunk_file)
                
                # Convert to WAV for API
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                    audio_segment.export(temp_file.name, format="wav")
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
                    print(f"   Voice Settings: {self.voice_settings}")
                    
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
    
    def process_chunks(self, input_dir, output_dir):
        """Process all chunks in input directory"""
        print(f"🎵 STS Chunk Streamer: Processing chunks from {input_dir}")
        print(f"🎵 Output directory: {output_dir}")
        print(f"🎵 Model: {self.model_id}")
        print(f"🎵 Voice: {self.voice_id}")
        print(f"🎵 Pro Features: {self.output_format}, {self.voice_settings}")
        print("=" * 60)
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Get all chunk files
        chunk_files = []
        for file in os.listdir(input_dir):
            if file.startswith("chunk_") and file.endswith(".mp3"):
                chunk_files.append(file)
        
        # Sort by chunk number
        chunk_files.sort(key=lambda x: int(x.split("_")[1].split(".")[0]))
        
        print(f"📁 Found {len(chunk_files)} chunks to process")
        
        processed_chunks = 0
        failed_chunks = 0
        
        for i, chunk_file in enumerate(chunk_files, 1):
            chunk_path = os.path.join(input_dir, chunk_file)
            
            print(f"\n🎵 Processing chunk {i}/{len(chunk_files)}: {chunk_file}")
            
            # Process chunk with STS API
            audio_output = self.process_chunk_with_sts(chunk_path, i)
            
            if audio_output:
                # Save output chunk
                output_filename = f"sts_chunk_{i}.mp3"
                output_path = os.path.join(output_dir, output_filename)
                
                with open(output_path, 'wb') as f:
                    f.write(audio_output)
                
                print(f"💾 Saved STS output: {output_filename} ({len(audio_output)} bytes)")
                processed_chunks += 1
            else:
                print(f"❌ Failed to process chunk {i}")
                failed_chunks += 1
        
        print(f"\n✅ Processing completed!")
        print(f"📊 Processed: {processed_chunks} chunks")
        print(f"❌ Failed: {failed_chunks} chunks")
        print(f"📁 Output saved to: {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Stream VAD chunks to ElevenLabs STS API")
    parser.add_argument("input_dir", help="Input directory containing VAD chunks")
    parser.add_argument("--output_dir", default="chunks_res", help="Output directory for STS chunks")
    args = parser.parse_args()
    
    if not os.path.exists(args.input_dir):
        print(f"❌ Input directory not found: {args.input_dir}")
        sys.exit(1)
    
    # Create and run the streamer
    streamer = STSChunkStreamer()
    streamer.process_chunks(args.input_dir, args.output_dir)

if __name__ == "__main__":
    main() 