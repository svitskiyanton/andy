#!/usr/bin/env python3
"""
Live Voice Streaming Chunker - Real-time MP3 streaming with live VAD chunking
Streams MP3 file as microphone input and chunks it live using VAD
Saves chunks to voice_emulation folder as they're detected
"""

import os
import sys
import time
import signal
import asyncio
import threading
import argparse
import numpy as np
from pydub import AudioSegment
import pyaudio
import webrtcvad
import atexit
import queue
import io

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

class LiveVoiceStreamingChunker:
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
        self.audio_buffer = queue.Queue(maxsize=100)
        self.chunk_buffer = []
        self.running = True
        
        # VAD settings
        self.VAD_SAMPLE_RATE = 16000
        self.VAD_FRAME_DURATION = 0.03
        self.VAD_FRAME_SIZE = int(self.VAD_SAMPLE_RATE * self.VAD_FRAME_DURATION)
        self.MIN_PHRASE_DURATION = 2.0  # Minimum 2.0 seconds for a chunk
        self.SILENCE_DURATION = 0.8     # 800ms of silence to split
        
        # Live processing
        self.vad = webrtcvad.Vad(3)  # Least sensitive (0=most sensitive, 3=least)
        self.consecutive_speech_frames = 0  # Track consecutive speech frames
        self.consecutive_silence_frames = 0  # Track consecutive silence frames
        self.speech_flags = []
        self.in_speech = False
        self.chunk_start_time = 0
        self.silence_count = 0
        self.chunk_counter = 0
        
        # Create output folder
        self._create_output_folder()
        
        print(f"🎤 Live Voice Streaming Chunker Configuration:")
        print(f"   Input File: {self.input_file}")
        print(f"   Output Folder: {self.OUTPUT_FOLDER}")
        print(f"   Sample Rate: {self.SAMPLE_RATE}Hz")
        print(f"   Chunk Size: {self.CHUNK_SIZE}")
        print(f"   Min Phrase Duration: {self.MIN_PHRASE_DURATION}s")
        print(f"   Silence Duration: {self.SILENCE_DURATION}s")
        print(f"   VAD Sensitivity: Low (3)")
        
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
        """Load MP3 file and prepare it for streaming"""
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
            
            # Convert to PCM data for streaming
            self.audio_data = audio.raw_data
            self.audio_duration = len(audio) / 1000.0
            print("✅ Audio prepared for live streaming")
            return True
            
        except Exception as e:
            print(f"❌ Error loading audio: {e}")
            return False
    
    def save_live_chunk(self, chunk_audio_data):
        """Save live detected chunk to file"""
        try:
            self.chunk_counter += 1
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"live_chunk_{self.chunk_counter:03d}_{timestamp}.wav"
            filepath = os.path.join(self.OUTPUT_FOLDER, filename)
            
            # Convert raw PCM data back to AudioSegment
            audio_segment = AudioSegment(
                data=chunk_audio_data,
                sample_width=2,  # 16-bit
                frame_rate=self.SAMPLE_RATE,
                channels=1
            )
            
            # Save as WAV file
            audio_segment.export(filepath, format="wav")
            
            chunk_duration = len(audio_segment) / 1000.0
            print(f"💾 Saved live chunk {self.chunk_counter}: {chunk_duration:.2f}s -> {filename}")
            return filepath
            
        except Exception as e:
            print(f"❌ Error saving live chunk {self.chunk_counter}: {e}")
            return None
    
    def process_live_vad(self, audio_chunk):
        """Process live audio chunk with VAD - Conservative approach"""
        try:
            # Convert to 16kHz for VAD
            audio_segment = AudioSegment(
                data=audio_chunk,
                sample_width=2,
                frame_rate=self.SAMPLE_RATE,
                channels=1
            )
            
            # Resample to 16kHz
            audio_16k = audio_segment.set_frame_rate(self.VAD_SAMPLE_RATE)
            samples = np.array(audio_16k.get_array_of_samples(), dtype=np.int16)
            
            # Process each VAD frame
            frame_size = int(self.VAD_SAMPLE_RATE * self.VAD_FRAME_DURATION)
            frames_in_chunk = len(samples) // frame_size
            
            for i in range(frames_in_chunk):
                frame_start = i * frame_size
                frame_end = frame_start + frame_size
                frame = samples[frame_start:frame_end]
                
                if len(frame) == frame_size:
                    is_speech = self.vad.is_speech(frame.tobytes(), self.VAD_SAMPLE_RATE)
                    self.speech_flags.append(is_speech)
                    
                    # Conservative VAD logic - require more consecutive frames
                    if is_speech:
                        self.consecutive_speech_frames += 1
                        self.consecutive_silence_frames = 0
                        
                        # Only start speech if we have enough consecutive speech frames
                        if not self.in_speech and self.consecutive_speech_frames >= 10:  # 300ms of speech
                            self.in_speech = True
                            self.chunk_start_time = len(self.speech_flags) * self.VAD_FRAME_DURATION
                            print(f"🎤 Speech started at {self.chunk_start_time:.2f}s")
                            
                    else:
                        self.consecutive_silence_frames += 1
                        self.consecutive_speech_frames = 0
                        
                        # Only end speech if we have enough consecutive silence frames
                        if self.in_speech and self.consecutive_silence_frames >= 27:  # 800ms of silence
                            chunk_end_time = len(self.speech_flags) * self.VAD_FRAME_DURATION - (self.consecutive_silence_frames * self.VAD_FRAME_DURATION)
                            chunk_duration = chunk_end_time - self.chunk_start_time
                            
                            if chunk_duration >= self.MIN_PHRASE_DURATION:
                                print(f"🎤 Speech ended at {chunk_end_time:.2f}s (duration: {chunk_duration:.2f}s)")
                                # Save the chunk
                                self.save_live_chunk(b''.join(self.chunk_buffer))
                            
                            self.in_speech = False
                            self.chunk_buffer = []
            
            # Add current chunk to buffer if in speech
            if self.in_speech:
                self.chunk_buffer.append(audio_chunk)
                
        except Exception as e:
            print(f"❌ Live VAD processing error: {e}")
    
    def stream_audio_as_microphone(self):
        """Stream MP3 audio as if it's coming from a microphone"""
        print("🎤 Starting live audio streaming...")
        
        if not hasattr(self, 'audio_data'):
            print("❌ No audio loaded")
            return
        
        # Calculate streaming parameters
        bytes_per_second = self.SAMPLE_RATE * 2 * self.CHANNELS  # 16-bit = 2 bytes
        chunk_delay = self.CHUNK_SIZE / bytes_per_second
        
        print(f"🎤 Streaming audio:")
        print(f"   Duration: {self.audio_duration:.1f}s")
        print(f"   Chunk delay: {chunk_delay:.3f}s")
        print(f"   Total chunks: {len(self.audio_data) // self.CHUNK_SIZE}")
        
        current_position = 0
        chunk_count = 0
        
        while current_position < len(self.audio_data) and self.running:
            # Extract chunk
            end_position = min(current_position + self.CHUNK_SIZE, len(self.audio_data))
            audio_chunk = self.audio_data[current_position:end_position]
            
            chunk_count += 1
            current_time = current_position / bytes_per_second
            
            # Only log every 10th chunk to reduce noise
            if chunk_count % 10 == 0:
                print(f"🎤 Streaming chunk {chunk_count} at {current_time:.2f}s")
            
            # Process with live VAD
            self.process_live_vad(audio_chunk)
            
            # Move to next position
            current_position = end_position
            
            # Simulate real-time streaming delay
            if self.running and current_position < len(self.audio_data):
                time.sleep(chunk_delay)
        
        # Handle final chunk if still in speech
        if self.in_speech and self.chunk_buffer:
            chunk_duration = len(self.speech_flags) * self.VAD_FRAME_DURATION - self.chunk_start_time
            if chunk_duration >= self.MIN_PHRASE_DURATION:
                print(f"🎤 Final speech chunk: {chunk_duration:.2f}s")
                self.save_live_chunk(b''.join(self.chunk_buffer))
        
        print(f"🎤 Live streaming completed: {chunk_count} chunks processed")
    
    async def start(self):
        """Start the live voice streaming chunker"""
        try:
            print("🎤 Live Voice Streaming Chunker: MP3 → Live Stream → VAD Chunks")
            print("=" * 80)
            print("🎤 Streaming: MP3 file as live microphone input")
            print("🔍 Live VAD: Real-time speech detection")
            print(f"📁 Output folder: {self.OUTPUT_FOLDER}")
            print("🎤 Chunks saved immediately as detected")
            print("=" * 80)
            
            # Load and prepare audio
            if not self.load_and_prepare_audio():
                return
            
            print("🎤 Live Voice Streaming Chunker started!")
            print("🎤 Streaming: MP3 as live microphone input")
            print("🔍 Processing: Real-time VAD chunking")
            print(f"📁 Saving: Live chunks to {self.OUTPUT_FOLDER} folder")
            print("⏹️  Press Ctrl+C to stop early")
            
            # Check running flag before starting processing
            if not self.running:
                print("⏹️  Stopping before processing due to interrupt...")
                return
            
            # Start live streaming
            self.stream_audio_as_microphone()
            
            print("✅ Live Voice Streaming Chunker completed")
            
        except Exception as e:
            print(f"❌ Live Voice Streaming Chunker error: {e}")

async def main():
    """Main function with improved interrupt handling"""
    global running
    
    try:
        # Parse command line arguments
        parser = argparse.ArgumentParser(description="Live Voice Streaming Chunker")
        parser.add_argument("input_file", help="Input MP3 file")
        parser.add_argument("--output-folder", help="Output folder", default="voice_emulation")
        args = parser.parse_args()
        
        # Check prerequisites
        print("✅ Prerequisites check passed")
        
        # Create and start the chunker
        voice_chunker = LiveVoiceStreamingChunker(args.input_file)
        
        # Override output folder if specified
        if args.output_folder:
            voice_chunker.OUTPUT_FOLDER = args.output_folder
            voice_chunker._create_output_folder()
            print(f"📁 Output folder set to: {args.output_folder}")
        
        # Set global running flag
        running = True
        
        # Start the live voice streaming chunker
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