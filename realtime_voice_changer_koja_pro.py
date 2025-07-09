#!/usr/bin/env python3
"""
Real-time Voice Changer using KojaB's RealtimeSTT + RealtimeTTS
Combines STT and TTS for seamless voice changing with ElevenLabs Pro Flash v2.5
"""

import asyncio
import json
import base64
import time
import numpy as np
import pyaudio
import threading
import queue
import os
import sys
from dotenv import load_dotenv
import websockets
from pydub import AudioSegment
import io

# Load environment variables
load_dotenv()

# Add KojaB libraries to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'KojaB', 'RealtimeSTT'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'KojaB', 'RealtimeTTS'))

try:
    from RealtimeSTT import AudioToTextRecorder
    from RealtimeTTS import TextToAudioStream, ElevenlabsEngine, ElevenlabsVoice
except ImportError as e:
    print(f"❌ Error importing KojaB libraries: {e}")
    print("Please ensure KojaB/RealtimeSTT and KojaB/RealtimeTTS are in the correct location")
    sys.exit(1)

class RealtimeVoiceChangerKojaPro:
    def __init__(self):
        # Audio settings
        self.FORMAT = pyaudio.paFloat32
        self.CHANNELS = 1
        self.SAMPLE_RATE = 44100
        self.CHUNK_SIZE = 2048  # Pro-level chunk size
        
        # Buffer settings (Pro limits)
        self.BUFFER_SIZE = 30
        self.MIN_BUFFER_SIZE = 10
        self.MAX_BUFFER_SIZE = 60
        
        # Audio objects
        self.audio = pyaudio.PyAudio()
        self.continuous_buffer = queue.Queue(maxsize=200)
        self.playback_thread = None
        
        # Playback control
        self.playback_started = False
        self.streaming_complete = False
        self.mp3_chunks = []
        
        # Output file for all TTS audio
        self.all_audio_chunks = []
        
        # Output file
        self.OUTPUT_FILE = "realtime_voice_changer_koja_pro.mp3"
        
        # In-memory queue for STT → TTS communication
        self.text_queue = queue.Queue(maxsize=20)
        
        # File logging for testing
        self.QUEUE_LOG_FILE = "queue_log_koja_pro.txt"
        self.OUTPUT_MP3_FILE = "realtime_voice_changer_koja_pro.mp3"
        
        # Control
        self.running = True
        
        # STT settings
        self.STT_SAMPLE_RATE = 44100  # Pro supports 44.1kHz
        self.STT_CHANNELS = 1
        self.STT_FORMAT = pyaudio.paInt16
        self.STT_CHUNK_SIZE = 2048
        
        # STT objects
        self.stt_audio = pyaudio.PyAudio()
        self.speech_client = None
        self.init_google_client()
        
        # STT buffers
        self.audio_buffer = []
        self.silence_buffer = []
        self.current_phrase = ""
        self.silence_start = None
        self.max_silence_duration = 1.0
        
        # STT settings
        self.BUFFER_DURATION = 2.0
        self.BUFFER_SIZE_STT = int(self.STT_SAMPLE_RATE * self.BUFFER_DURATION)
        self.SILENCE_DURATION = 0.5
        self.SILENCE_THRESHOLD = 0.008
        
        # Text processing
        self.text_processor_thread = None
        
        # Initialize KojaB components
        self.init_koja_components()
        
    def init_google_client(self):
        """Initialize Google Cloud Speech client"""
        try:
            from google.cloud import speech
            credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
            if credentials_path:
                self.speech_client = speech.SpeechClient()
                print("✅ STT: Google Cloud Speech client initialized")
            else:
                print("⚠️  STT: Google Cloud credentials not found")
        except Exception as e:
            print(f"❌ STT: Failed to initialize Google Cloud client: {e}")
    
    def init_koja_components(self):
        """Initialize KojaB STT and TTS components"""
        try:
            # Initialize STT recorder
            self.stt_recorder = AudioToTextRecorder(
                device="default",
                language="ru-RU",
                model="base",  # Use base model for speed
                energy_threshold=1000,
                record_timeout=1.0,
                phrase_timeout=3.0,
                phrase_threshold=0.3,
                non_speaking_duration=0.5,
                enable_silence_detection=True
            )
            print("✅ KojaB STT: AudioToTextRecorder initialized")
            
            # Initialize TTS engine with ElevenLabs Pro Flash v2.5
            api_key = self._get_api_key()
            if api_key:
                self.tts_engine = ElevenlabsEngine(
                    api_key=api_key,
                    voice="Nicole",  # Default voice
                    id="piTKgcLEGmPE4e6mEKli",
                    model="eleven_flash_v2_5",  # Pro Flash v2.5 model
                    clarity=75.0,
                    stability=50.0,
                    style_exxageration=0.0
                )
                print("✅ KojaB TTS: ElevenLabs Flash v2.5 engine initialized")
                
                # Initialize TTS stream
                self.tts_stream = TextToAudioStream(
                    engine=self.tts_engine,
                    on_audio_chunk=self._on_audio_chunk,
                    on_playback_start=self._on_playback_start,
                    on_playback_end=self._on_playback_end
                )
                print("✅ KojaB TTS: TextToAudioStream initialized")
            else:
                print("❌ TTS: ElevenLabs API key not found")
                
        except Exception as e:
            print(f"❌ Error initializing KojaB components: {e}")
    
    def _get_api_key(self):
        """Get API key from environment, file, or user input"""
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
    
    def _on_audio_chunk(self, chunk):
        """Callback for TTS audio chunks"""
        try:
            if chunk:
                # Add to continuous buffer for playback
                try:
                    self.continuous_buffer.put_nowait(chunk)
                except queue.Full:
                    pass
                
                # Collect for file saving
                self.all_audio_chunks.append(chunk)
                
                print(f"🎵 TTS: Audio chunk received ({len(chunk)} bytes)")
                
        except Exception as e:
            print(f"❌ Error processing audio chunk: {e}")
    
    def _on_playback_start(self):
        """Callback when TTS playback starts"""
        print("🎵 TTS: Playback started")
        self.playback_started = True
    
    def _on_playback_end(self):
        """Callback when TTS playback ends"""
        print("🎵 TTS: Playback ended")
        self.playback_started = False
    
    def add_text_to_queue(self, phrase):
        """Add transcribed text to in-memory queue for TTS processing"""
        try:
            self.text_queue.put_nowait(phrase)
            print(f"📝 STT → TTS Queue: '{phrase}'")
            self.log_to_file(f"ADDED: {phrase}")
        except queue.Full:
            try:
                old_phrase = self.text_queue.get_nowait()
                self.text_queue.put_nowait(phrase)
                print(f"🔄 STT → TTS Queue: Replaced '{old_phrase}' with '{phrase}'")
                self.log_to_file(f"REPLACED: '{old_phrase}' → '{phrase}'")
            except Exception as e:
                print(f"❌ Error managing queue: {e}")
        except Exception as e:
            print(f"❌ Error adding to queue: {e}")
    
    def log_to_file(self, message):
        """Log queue operations to file for testing"""
        try:
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            with open(self.QUEUE_LOG_FILE, 'a', encoding='utf-8') as f:
                f.write(f"[{timestamp}] {message}\n")
        except Exception as e:
            print(f"❌ Error logging to file: {e}")
    
    def stt_worker(self):
        """STT worker using KojaB's AudioToTextRecorder"""
        print("🎤 STT Worker: Starting KojaB AudioToTextRecorder...")
        
        try:
            # Use KojaB's STT recorder
            while self.running:
                try:
                    # Get text from KojaB STT
                    text = self.stt_recorder.text()
                    if text and text.strip():
                        print(f"📝 STT: '{text}'")
                        self.add_text_to_queue(text.strip())
                    
                    time.sleep(0.01)  # Small delay
                    
                except Exception as e:
                    print(f"❌ STT: Error in audio processing: {e}")
                    break
            
            print("✅ STT Worker stopped")
            
        except Exception as e:
            print(f"❌ STT Worker error: {e}")
    
    def text_processor_worker(self):
        """Process text from in-memory queue and trigger TTS"""
        print("📝 Text Processor: Monitoring in-memory queue...")
        
        while self.running:
            try:
                try:
                    text = self.text_queue.get(timeout=0.1)
                    print(f"🎵 TTS: Processing text from queue: '{text}'")
                    self.log_to_file(f"PROCESSING: {text}")
                    
                    # Use KojaB's TTS stream
                    if hasattr(self, 'tts_stream'):
                        self.tts_stream.feed_text(text)
                    
                    self.text_queue.task_done()
                    
                except queue.Empty:
                    continue
                    
            except Exception as e:
                print(f"❌ Text processor error: {e}")
                time.sleep(0.1)
    
    def continuous_playback_worker(self):
        """Continuous playback worker that eliminates gaps"""
        try:
            # Initialize audio stream
            stream = self.audio.open(
                format=self.FORMAT,
                channels=self.CHANNELS,
                rate=self.SAMPLE_RATE,
                output=True,
                frames_per_buffer=self.CHUNK_SIZE
            )
            
            print("🎵 Continuous playback worker started")
            
            while self.running:
                try:
                    # Get audio chunk (blocking with timeout)
                    audio_data = self.continuous_buffer.get(timeout=0.1)
                    
                    # Play audio immediately
                    if isinstance(audio_data, bytes):
                        # Convert bytes to float32 for playback
                        audio_array = np.frombuffer(audio_data, dtype=np.float32)
                        stream.write(audio_array.tobytes())
                    
                except queue.Empty:
                    continue
                except Exception as e:
                    print(f"⚠️ Playback error: {e}")
                    continue
            
            stream.stop_stream()
            stream.close()
            print("🎵 Continuous playback worker completed")
            
        except Exception as e:
            print(f"❌ Continuous playback worker error: {e}")
    
    def start(self):
        """Start the real-time voice changer"""
        print("🎤🎵 Real-time Voice Changer (KojaB + ElevenLabs Pro)")
        print("=" * 60)
        print("🎤 STT: KojaB AudioToTextRecorder")
        print("📝 Text Processor: Processes from memory queue")
        print("🎵 TTS: KojaB TextToAudioStream + ElevenLabs Flash v2.5")
        print(f"📁 Queue log: {self.QUEUE_LOG_FILE}")
        print(f"🎵 Output file: {self.OUTPUT_MP3_FILE}")
        print("=" * 60)
        
        # Clear log file
        if os.path.exists(self.QUEUE_LOG_FILE):
            os.remove(self.QUEUE_LOG_FILE)
        
        # Start STT worker thread
        stt_thread = threading.Thread(target=self.stt_worker)
        stt_thread.daemon = True
        stt_thread.start()
        
        # Start text processor thread
        self.text_processor_thread = threading.Thread(target=self.text_processor_worker)
        self.text_processor_thread.daemon = True
        self.text_processor_thread.start()
        
        # Start continuous playback thread
        playback_thread = threading.Thread(target=self.continuous_playback_worker)
        playback_thread.daemon = True
        playback_thread.start()
        
        print("✅ Real-time voice changer (KojaB Pro) started!")
        print("🎤 Speak into your microphone...")
        print("🎵 Your voice will be changed in real-time!")
        print("⏹️  Press Ctrl+C to stop")
        
        try:
            # Keep main thread alive
            while self.running:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n⏹️  Stopping real-time voice changer...")
            self.running = False
            self.save_final_output()
            print("✅ Real-time voice changer stopped")
    
    def save_final_output(self):
        """Save all TTS audio to a single MP3 file"""
        try:
            if self.all_audio_chunks:
                print(f"💾 Saving {len(self.all_audio_chunks)} audio chunks to {self.OUTPUT_MP3_FILE}...")
                
                # Combine all audio chunks
                combined_audio = b''.join(self.all_audio_chunks)
                
                # Save as MP3
                with open(self.OUTPUT_MP3_FILE, 'wb') as f:
                    f.write(combined_audio)
                
                print(f"✅ All audio saved to: {self.OUTPUT_MP3_FILE}")
                print(f"📊 Total audio size: {len(combined_audio)} bytes")
                
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
    # Check prerequisites
    if not os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
        print("❌ GOOGLE_APPLICATION_CREDENTIALS not found")
        print("   Please set the path to your Google Cloud service account key")
        return
    
    print("✅ Prerequisites check passed")
    
    # Create and start voice changer
    voice_changer = RealtimeVoiceChangerKojaPro()
    voice_changer.start()

if __name__ == "__main__":
    asyncio.run(main()) 