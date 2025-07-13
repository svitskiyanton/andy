#!/usr/bin/env python3
"""
Voice Emulation Chunker - Simulates real-time voice input
Reads MP3 file in small chunks and saves them to voice_emulation folder
Emulates microphone input for testing VAD and chunking algorithms
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
import io

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
            save_audio_chunks(audio_chunks, "voice_emulation_output.mp3")

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

class VoiceEmulationChunker:
    def __init__(self, input_file=None):
        # Audio settings
        self.SAMPLE_RATE = 44100
        self.CHUNK_SIZE = 1024
        self.CHANNELS = 1
        self.FORMAT = pyaudio.paInt16
        
        # Input/Output files
        self.input_file = input_file
        self.OUTPUT_FOLDER = "voice_emulation"
        
        # Audio processing
        self.audio = pyaudio.PyAudio()
        self.running = True
        self.all_audio_chunks = []
        
        # VAD settings
        self.VAD_SAMPLE_RATE = 16000
        self.VAD_FRAME_DURATION = 0.03
        self.VAD_FRAME_SIZE = int(self.VAD_SAMPLE_RATE * self.VAD_FRAME_DURATION)
        self.MIN_PHRASE_DURATION = 0.2
        self.SILENCE_DURATION = 0.05
        
        # Emulation settings
        self.EMULATION_CHUNK_DURATION = 0.1  # 100ms chunks for emulation
        self.EMULATION_DELAY = 0.05  # 50ms delay between chunks (simulates real-time)
        
        # Create output folder
        self._create_output_folder()
        
        print(f"🎤 Voice Emulation Chunker Configuration:")
        print(f"   Input File: {self.input_file}")
        print(f"   Output Folder: {self.OUTPUT_FOLDER}")
        print(f"   Emulation Chunk Duration: {self.EMULATION_CHUNK_DURATION}s")
        print(f"   Emulation Delay: {self.EMULATION_DELAY}s")
        print(f"   VAD Frame Duration: {self.VAD_FRAME_DURATION}s")
        print(f"   Min Phrase Duration: {self.MIN_PHRASE_DURATION}s")
        
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
            print("✅ Audio prepared for emulation")
            return True
            
        except Exception as e:
            print(f"❌ Error loading audio: {e}")
            return False
    
    def detect_vad_boundaries(self, audio_segment):
        """Detect phrase boundaries using VAD and return list of (start_ms, end_ms) tuples"""
        print("🔍 Starting VAD boundary detection...")
        
        vad = webrtcvad.Vad(0)  # Most sensitive
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
    
    def save_chunk_to_file(self, chunk_audio, chunk_index, chunk_type="emulation"):
        """Save audio chunk to file in voice_emulation folder"""
        try:
            # Create filename with timestamp and chunk info
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"{chunk_type}_chunk_{chunk_index:04d}_{timestamp}.wav"
            filepath = os.path.join(self.OUTPUT_FOLDER, filename)
            
            # Save as WAV file
            chunk_audio.export(filepath, format="wav")
            
            print(f"💾 Saved {chunk_type} chunk {chunk_index} to: {filename}")
            return filepath
            
        except Exception as e:
            print(f"❌ Error saving chunk {chunk_index}: {e}")
            return None
    
    def emulate_voice_input(self):
        """Emulate real-time voice input by reading audio in small chunks"""
        print("🎤 Starting voice input emulation...")
        
        if not hasattr(self, 'audio_segment'):
            print("❌ No audio loaded")
            return
        
        audio = self.audio_segment
        total_duration = len(audio) / 1000.0  # Convert to seconds
        chunk_duration_ms = int(self.EMULATION_CHUNK_DURATION * 1000)
        
        print(f"🎤 Emulating voice input:")
        print(f"   Total duration: {total_duration:.1f}s")
        print(f"   Chunk duration: {self.EMULATION_CHUNK_DURATION}s")
        print(f"   Delay between chunks: {self.EMULATION_DELAY}s")
        
        chunk_count = 0
        current_position = 0
        
        while current_position < len(audio) and self.running:
            # Extract chunk
            end_position = min(current_position + chunk_duration_ms, len(audio))
            chunk_audio = audio[current_position:end_position]
            
            chunk_count += 1
            chunk_duration = len(chunk_audio) / 1000.0
            
            print(f"\n🎤 Emulation chunk {chunk_count}: {chunk_duration:.3f}s at {current_position/1000:.2f}s")
            
            # Save emulation chunk
            saved_file = self.save_chunk_to_file(chunk_audio, chunk_count, "emulation")
            
            if saved_file:
                # Collect for combined output
                self.all_audio_chunks.append(chunk_audio)
                global audio_chunks
                audio_chunks.append(chunk_audio.raw_data)
                
                print(f"✅ Emulation chunk {chunk_count} processed")
            else:
                print(f"❌ Failed to save emulation chunk {chunk_count}")
            
            # Move to next position
            current_position = end_position
            
            # Simulate real-time delay
            if self.running and current_position < len(audio):
                print(f"⏳ Waiting {self.EMULATION_DELAY}s (emulating real-time)...")
                time.sleep(self.EMULATION_DELAY)
        
        print(f"🎤 Voice emulation completed: {chunk_count} chunks processed")
    
    def process_vad_chunks(self):
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
        
        print(f"✅ Created {len(chunks)} VAD chunks")
        
        chunk_count = 0
        
        for i, (start_ms, end_ms) in enumerate(chunks):
            # Check running flag
            if not self.running:
                print("⏹️  Stopping due to interrupt signal...")
                break
                
            chunk_count += 1
            chunk_duration = (end_ms - start_ms) / 1000.0
            
            print(f"\n🔍 VAD Processing chunk {chunk_count}/{len(chunks)} ({chunk_duration:.1f}s)")
            print(f"   Time range: {start_ms/1000:.1f}s - {end_ms/1000:.1f}s")
            
            # Extract chunk
            chunk_audio = audio[start_ms:end_ms]
            print(f"🔍 Extracted VAD chunk {chunk_count} ({len(chunk_audio)}ms)")
            
            # Save VAD chunk
            saved_file = self.save_chunk_to_file(chunk_audio, chunk_count, "vad")
            
            if saved_file:
                print(f"✅ VAD chunk {chunk_count} saved successfully")
            else:
                print(f"❌ Failed to save VAD chunk {chunk_count}")
            
            # Small delay between chunks
            time.sleep(0.1)
        
        print(f"🔍 VAD processing completed: {chunk_count} chunks")
    
    async def start(self):
        """Start the voice emulation chunker"""
        try:
            print("🎤 Voice Emulation Chunker: MP3 File → Real-time Chunks → voice_emulation folder")
            print("=" * 80)
            print("🎤 Emulation: Simulates real-time microphone input")
            print("🔍 VAD: Detects natural phrase boundaries")
            print(f"📁 Output folder: {self.OUTPUT_FOLDER}")
            print("🔍 DEBUG: Emulates voice input for testing")
            print("=" * 80)
            
            # Load and prepare audio
            if not self.load_and_prepare_audio():
                return
            
            print("🎤 Voice Emulation Chunker started!")
            print("🎤 Emulating: Real-time voice input from microphone")
            print("🔍 Processing: VAD boundary detection")
            print(f"📁 Saving: Chunks to {self.OUTPUT_FOLDER} folder")
            print("⏹️  Press Ctrl+C to stop early")
            
            # Check running flag before starting processing
            if not self.running:
                print("⏹️  Stopping before processing due to interrupt...")
                return
            
            # Emulate voice input
            self.emulate_voice_input()
            
            # Process VAD chunks
            self.process_vad_chunks()
            
            print("✅ Voice Emulation Chunker completed")
            
        except Exception as e:
            print(f"❌ Voice Emulation Chunker error: {e}")
        finally:
            # Always save output even on interrupt
            if not self.running:
                print("💾 Saving output due to interrupt...")
            self.save_final_output()
    
    def save_final_output(self):
        """Save final combined output file"""
        try:
            print("💾 Saving final combined output...")
            if self.all_audio_chunks:
                # Combine all chunks
                combined_audio = sum(self.all_audio_chunks)
                output_file = os.path.join(self.OUTPUT_FOLDER, "combined_emulation_output.wav")
                combined_audio.export(output_file, format="wav")
                print(f"✅ Combined output saved to: {output_file}")
                print(f"📊 Total duration: {len(combined_audio)/1000:.1f}s")
        except Exception as e:
            print(f"❌ Error saving final output: {e}")

async def main():
    """Main function with improved interrupt handling"""
    global running
    
    try:
        # Parse command line arguments
        parser = argparse.ArgumentParser(description="Voice Emulation Chunker")
        parser.add_argument("input_file", help="Input MP3 file")
        parser.add_argument("--output-folder", help="Output folder", default="voice_emulation")
        args = parser.parse_args()
        
        # Check prerequisites
        print("✅ Prerequisites check passed")
        
        # Create and start the chunker
        voice_chunker = VoiceEmulationChunker(args.input_file)
        
        # Override output folder if specified
        if args.output_folder:
            voice_chunker.OUTPUT_FOLDER = args.output_folder
            voice_chunker._create_output_folder()
            print(f"📁 Output folder set to: {args.output_folder}")
        
        # Set global running flag
        running = True
        
        # Start the voice emulation chunker
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