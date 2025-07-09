#!/usr/bin/env python3
"""
Single Chunk Voice Changer Test - FIXED VERSION
Using official ElevenLabs API configuration to fix speech impediment issues
"""

import os
import sys
import time
import signal
from dotenv import load_dotenv
from elevenlabs.client import ElevenLabs
from elevenlabs import play
import pyaudio
import wave
import io

class SingleChunkTestFixed:
    def __init__(self, api_key: str, voice_id: str):
        """
        Initialize the single chunk test with correct API configuration
        
        Args:
            api_key: ElevenLabs API key
            voice_id: Target voice ID for transformation
        """
        self.api_key = api_key
        self.voice_id = voice_id
        
        # Initialize ElevenLabs client properly
        self.elevenlabs = ElevenLabs(api_key=api_key)
        
        # Audio settings
        self.audio_format = pyaudio.paInt16
        self.channels = 1
        self.rate = 16000
        self.chunk_size = 2048
        self.test_duration = 3.0  # Exactly 3 seconds
        
        # Audio objects
        self.pyaudio = None
        self.input_stream = None
        
    def setup_audio(self):
        """Setup audio input stream"""
        try:
            self.pyaudio = pyaudio.PyAudio()
            
            # Get default devices
            input_device = self.pyaudio.get_default_input_device_info()
            
            print(f"🎤 Input Device: {input_device['name']}")
            print(f"📊 Sample Rate: {self.rate}Hz, Channels: {self.channels}")
            print(f"📦 Chunk Size: {self.chunk_size} samples ({self.chunk_size/self.rate*1000:.0f}ms)")
            print(f"⏱️ Test Duration: {self.test_duration}s")
            print(f"🇷🇺 Language: Russian")
            print(f"🎯 Voice ID: {self.voice_id}")
            
            # Setup input stream
            self.input_stream = self.pyaudio.open(
                format=self.audio_format,
                channels=self.channels,
                rate=self.rate,
                input=True,
                frames_per_buffer=self.chunk_size
            )
            
            print("✅ Audio setup complete")
            
        except Exception as e:
            print(f"❌ Audio setup failed: {e}")
            sys.exit(1)
    
    def process_audio_chunk(self, audio_data):
        """Process audio chunk using official ElevenLabs API"""
        print("🔄 Processing audio through ElevenLabs API...")
        
        try:
            # Convert PCM to WAV bytes
            wav_data = self.pcm_to_wav_bytes(audio_data)
            
            # Create BytesIO object for the API
            audio_stream = io.BytesIO(wav_data)
            
            print("📤 Sending to ElevenLabs API...")
            start_time = time.time()
            
            # Use the official speech_to_speech.convert method
            # Based on ElevenLabs documentation example
            audio_stream_result = self.elevenlabs.speech_to_speech.convert(
                voice_id=self.voice_id,
                audio=audio_stream,
                model_id="eleven_multilingual_sts_v2",
                output_format="mp3_44100_128",  # Use MP3 format as per docs
            )
            
            # Save the result
            output_filename = "single_chunk_result_fixed.wav"
            
            # Convert MP3 to WAV for easier playback
            with open(output_filename, 'wb') as f:
                f.write(audio_stream_result)
            
            end_time = time.time()
            processing_time = (end_time - start_time) * 1000
            
            print(f"✅ Processing completed in {processing_time:.0f}ms")
            print(f"✅ Result saved to: {output_filename}")
            return output_filename
                
        except Exception as e:
            print(f"❌ Error processing audio: {e}")
            return None
    
    def pcm_to_wav_bytes(self, pcm_data, sample_rate=16000, channels=1, sample_width=2):
        """Convert PCM data to WAV format"""
        buffer = io.BytesIO()
        with wave.open(buffer, 'wb') as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(sample_width)
            wf.setframerate(sample_rate)
            wf.writeframes(pcm_data)
        return buffer.getvalue()
    
    def capture_and_process(self):
        """Capture exactly 3 seconds of audio and process it"""
        print(f"\n🎤 Capturing {self.test_duration}s of audio...")
        print("🎯 Speak now! Press Enter to start recording...")
        
        # Wait for user to start
        input()
        
        print("🔴 Recording... (3 seconds)")
        
        # Capture audio for exactly 3 seconds
        audio_chunks = []
        start_time = time.time()
        
        while time.time() - start_time < self.test_duration:
            try:
                audio_data = self.input_stream.read(self.chunk_size, exception_on_overflow=False)
                audio_chunks.append(audio_data)
            except Exception as e:
                print(f"⚠️ Audio capture error: {e}")
                break
        
        # Combine all chunks
        combined_audio = b''.join(audio_chunks)
        actual_duration = len(audio_chunks) * self.chunk_size / self.rate
        print(f"📦 Captured {actual_duration:.1f}s of audio")
        
        # Save original audio for comparison
        original_filename = "single_chunk_original_fixed.wav"
        with wave.open(original_filename, 'wb') as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(self.rate)
            wf.writeframes(combined_audio)
        
        print(f"📁 Original audio saved to: {original_filename}")
        
        # Process the chunk
        result_file = self.process_audio_chunk(combined_audio)
        
        return result_file
    
    def run_test(self):
        """Run the single chunk test"""
        print("🧪 Single Chunk Voice Changer Test - FIXED VERSION")
        print("=" * 60)
        print("🔧 Using official ElevenLabs API configuration")
        print("=" * 60)
        
        try:
            # Setup audio
            self.setup_audio()
            
            # Capture and process
            result = self.capture_and_process()
            
            # Summary
            print(f"\n{'='*60}")
            print("📊 Test Results:")
            print(f"{'='*60}")
            
            if result:
                print(f"✅ Success: {result}")
                print(f"📁 Original: single_chunk_original_fixed.wav")
                print(f"📁 Transformed: {result}")
                print(f"\n🎵 You can now compare the original and transformed audio!")
                print(f"🔧 This version uses the official ElevenLabs API configuration")
            else:
                print(f"❌ Test failed")
            
        except KeyboardInterrupt:
            print("\n🛑 Test interrupted by user")
        except Exception as e:
            print(f"❌ Test error: {e}")
        finally:
            if self.input_stream:
                self.input_stream.stop_stream()
                self.input_stream.close()
            if self.pyaudio:
                self.pyaudio.terminate()

def signal_handler(signum, frame):
    """Handle Ctrl+C gracefully"""
    print("\n🛑 Received interrupt signal")
    sys.exit(0)

def main():
    """Main function"""
    # Load environment variables
    load_dotenv()
    
    # Get API credentials
    api_key = os.getenv("ELEVENLABS_API_KEY")
    voice_id = os.getenv("VOICE_ID")
    
    if not api_key or api_key == "your_api_key_here":
        print("❌ Please set your ELEVENLABS_API_KEY in the .env file")
        print("📝 Copy env_template.txt to .env and add your API key")
        sys.exit(1)
    
    if not voice_id or voice_id == "your_voice_id_here":
        print("❌ Please set your VOICE_ID in the .env file")
        print("📝 You can find voice IDs in your ElevenLabs dashboard")
        sys.exit(1)
    
    # Setup signal handler
    signal.signal(signal.SIGINT, signal_handler)
    
    # Create and run test
    tester = SingleChunkTestFixed(api_key, voice_id)
    tester.run_test()

if __name__ == "__main__":
    main() 