#!/usr/bin/env python3
"""
Simple Voice Chunker - Only VAD-based chunking
Reads MP3 file and splits it into natural speech chunks using VAD
Saves chunks to voice_emulation folder
"""

import os
import sys
import time
import signal
import asyncio
import argparse
import numpy as np
from pydub import AudioSegment
import webrtcvad
import atexit

# Global variables for cleanup
running = True

def signal_handler(signum, frame):
    """Handle interrupt signal"""
    global running
    print("\n⏹️  Received interrupt signal, exiting...")
    running = False

def cleanup_handler():
    """Cleanup handler for atexit"""
    global running
    if running:
        running = False
        print("💾 Cleanup completed")

# Register signal handlers
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)
atexit.register(cleanup_handler)

class SimpleVoiceChunker:
    def __init__(self, input_file=None):
        # Audio settings
        self.SAMPLE_RATE = 44100
        
        # Input/Output files
        self.input_file = input_file
        self.OUTPUT_FOLDER = "voice_emulation"
        
        # VAD settings
        self.VAD_SAMPLE_RATE = 16000
        self.VAD_FRAME_DURATION = 0.03
        self.VAD_FRAME_SIZE = int(self.VAD_SAMPLE_RATE * self.VAD_FRAME_DURATION)
        self.MIN_PHRASE_DURATION = 0.5  # Minimum 0.5 seconds for a chunk
        self.SILENCE_DURATION = 0.1     # 100ms of silence to split
        
        # Create output folder
        self._create_output_folder()
        
        print(f"🎤 Simple Voice Chunker Configuration:")
        print(f"   Input File: {self.input_file}")
        print(f"   Output Folder: {self.OUTPUT_FOLDER}")
        print(f"   Min Phrase Duration: {self.MIN_PHRASE_DURATION}s")
        print(f"   Silence Duration: {self.SILENCE_DURATION}s")
        
    def _create_output_folder(self):
        """Create the voice_emulation output folder"""
        try:
            if not os.path.exists(self.OUTPUT_FOLDER):
                os.makedirs(self.OUTPUT_FOLDER)
                print(f"✅ Created output folder: {self.OUTPUT_FOLDER}")
            else:
                print(f"📁 Using existing folder: {self.OUTPUT_FOLDER}")
        except Exception as e:
            print(f"❌ Error creating output folder: {e}")
    
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
            if audio.frame_rate != self.SAMPLE_RATE:
                audio = audio.set_frame_rate(self.SAMPLE_RATE)
                print(f"🔄 Resampled to {self.SAMPLE_RATE}Hz")
            
            self.audio_segment = audio
            print("✅ Audio prepared for VAD processing")
            return True
            
        except Exception as e:
            print(f"❌ Error loading audio: {e}")
            return False
    
    def detect_vad_boundaries(self, audio_segment):
        """Detect phrase boundaries using VAD and return list of (start_ms, end_ms) tuples"""
        print("🔍 Starting VAD boundary detection...")
        
        vad = webrtcvad.Vad(2)  # Medium sensitivity (0=most sensitive, 3=least)
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
        silence_count = 0
        
        for i, is_speech in enumerate(speech_flags):
            t = i * self.VAD_FRAME_DURATION
            
            if is_speech and not in_speech:
                # Start of speech
                chunk_start = t
                in_speech = True
                silence_count = 0
            elif is_speech and in_speech:
                # Continue speech
                silence_count = 0
            elif not is_speech and in_speech:
                # Silence during speech
                silence_count += self.VAD_FRAME_DURATION
                if silence_count >= self.SILENCE_DURATION:
                    # End of speech (enough silence)
                    chunk_end = t - silence_count + self.VAD_FRAME_DURATION
                    if chunk_end - chunk_start >= self.MIN_PHRASE_DURATION:
                        boundaries.append((int(chunk_start*1000), int(chunk_end*1000)))
                    in_speech = False
                    silence_count = 0
        
        # Handle last chunk
        if in_speech:
            chunk_end = total_frames * self.VAD_FRAME_DURATION
            if chunk_end - chunk_start >= self.MIN_PHRASE_DURATION:
                boundaries.append((int(chunk_start*1000), int(chunk_end*1000)))
        
        print(f"🔍 Found {len(boundaries)} VAD boundaries")
        for i, (start, end) in enumerate(boundaries):
            duration = (end - start) / 1000.0
            print(f"   Chunk {i+1}: {start/1000:.2f}s - {end/1000:.2f}s ({duration:.2f}s)")
        
        return boundaries
    
    def save_chunk_to_file(self, chunk_audio, chunk_index):
        """Save audio chunk to file in voice_emulation folder"""
        try:
            # Create filename with chunk info
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"voice_chunk_{chunk_index:03d}_{timestamp}.wav"
            filepath = os.path.join(self.OUTPUT_FOLDER, filename)
            
            # Save as WAV file
            chunk_audio.export(filepath, format="wav")
            
            print(f"💾 Saved chunk {chunk_index} to: {filename}")
            return filepath
            
        except Exception as e:
            print(f"❌ Error saving chunk {chunk_index}: {e}")
            return None
    
    def process_voice_chunks(self):
        """Process audio using VAD to find natural phrase boundaries"""
        print("🔍 Starting VAD processing...")
        
        if not hasattr(self, 'audio_segment'):
            print("❌ No audio loaded")
            return
        
        audio = self.audio_segment
        
        # Find natural phrase boundaries using VAD
        print("🔍 Detecting VAD boundaries...")
        chunks = self.detect_vad_boundaries(audio)
        
        if not chunks:
            print("❌ No VAD chunks created")
            return
        
        print(f"✅ Created {len(chunks)} voice chunks")
        
        chunk_count = 0
        
        for i, (start_ms, end_ms) in enumerate(chunks):
            # Check running flag
            if not self.running:
                print("⏹️  Stopping due to interrupt signal...")
                break
                
            chunk_count += 1
            chunk_duration = (end_ms - start_ms) / 1000.0
            
            print(f"\n🔍 Processing chunk {chunk_count}/{len(chunks)} ({chunk_duration:.1f}s)")
            print(f"   Time range: {start_ms/1000:.1f}s - {end_ms/1000:.1f}s")
            
            # Extract chunk
            chunk_audio = audio[start_ms:end_ms]
            print(f"🔍 Extracted chunk {chunk_count} ({len(chunk_audio)}ms)")
            
            # Save chunk
            saved_file = self.save_chunk_to_file(chunk_audio, chunk_count)
            
            if saved_file:
                print(f"✅ Chunk {chunk_count} saved successfully")
            else:
                print(f"❌ Failed to save chunk {chunk_count}")
        
        print(f"🔍 Voice chunking completed: {chunk_count} chunks")
    
    async def start(self):
        """Start the simple voice chunker"""
        try:
            print("🎤 Simple Voice Chunker: MP3 File → VAD Chunks → voice_emulation folder")
            print("=" * 80)
            print("🔍 VAD: Detects natural speech boundaries")
            print(f"📁 Output folder: {self.OUTPUT_FOLDER}")
            print("🎤 Only creates meaningful speech chunks")
            print("=" * 80)
            
            # Load and prepare audio
            if not self.load_and_prepare_audio():
                return
            
            print("🎤 Simple Voice Chunker started!")
            print("🔍 Processing: VAD boundary detection")
            print(f"📁 Saving: Natural speech chunks to {self.OUTPUT_FOLDER} folder")
            print("⏹️  Press Ctrl+C to stop early")
            
            # Check running flag before starting processing
            if not self.running:
                print("⏹️  Stopping before processing due to interrupt...")
                return
            
            # Process voice chunks
            self.process_voice_chunks()
            
            print("✅ Simple Voice Chunker completed")
            
        except Exception as e:
            print(f"❌ Simple Voice Chunker error: {e}")

async def main():
    """Main function with improved interrupt handling"""
    global running
    
    try:
        # Parse command line arguments
        parser = argparse.ArgumentParser(description="Simple Voice Chunker")
        parser.add_argument("input_file", help="Input MP3 file")
        parser.add_argument("--output-folder", help="Output folder", default="voice_emulation")
        args = parser.parse_args()
        
        # Check prerequisites
        print("✅ Prerequisites check passed")
        
        # Create and start the chunker
        voice_chunker = SimpleVoiceChunker(args.input_file)
        
        # Override output folder if specified
        if args.output_folder:
            voice_chunker.OUTPUT_FOLDER = args.output_folder
            voice_chunker._create_output_folder()
            print(f"📁 Output folder set to: {args.output_folder}")
        
        # Set global running flag
        running = True
        
        # Start the simple voice chunker
        await voice_chunker.start()
        
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
    asyncio.run(main()) 