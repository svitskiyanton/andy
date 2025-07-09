#!/usr/bin/env python3
"""
Real-time Voice Changer with Smooth Live Playback
Combines STT Writer + Smooth Live Playback for gap-free voice changing
"""

import asyncio
import json
import base64
import time
import numpy as np
import pyaudio
from collections import deque
import threading
import queue
from pydub import AudioSegment
import io
import os
import subprocess
import sys
from google.cloud import speech
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class RealtimeVoiceChangerSmooth:
    def __init__(self):
        # Audio settings
        self.FORMAT = pyaudio.paFloat32
        self.CHANNELS = 1
        self.SAMPLE_RATE = 44100
        self.CHUNK_SIZE = 1024
        
        # Buffer settings
        self.BUFFER_SIZE = 15
        self.MIN_BUFFER_SIZE = 5
        self.MAX_BUFFER_SIZE = 30
        
        # Audio objects
        self.audio = pyaudio.PyAudio()
        self.continuous_buffer = queue.Queue(maxsize=100)
        self.playback_thread = None
        
        # Playback control
        self.playback_started = False
        self.streaming_complete = False
        self.mp3_chunks = []
        
        # Output file for all TTS audio
        self.all_audio_chunks = []
        
        # Output file
        self.OUTPUT_FILE = "realtime_voice_changer_output.mp3"
        
        # In-memory LIFO queue for STT → TTS communication
        self.text_queue = queue.Queue(maxsize=10)  # LIFO-like behavior with max 10 items
        
        # File logging for testing
        self.QUEUE_LOG_FILE = "queue_log.txt"
        self.OUTPUT_MP3_FILE = "realtime_voice_changer_output.mp3"
        
        # Control
        self.running = True
        
        # STT settings
        self.STT_SAMPLE_RATE = 16000
        self.STT_CHANNELS = 1
        self.STT_FORMAT = pyaudio.paInt16
        self.STT_CHUNK_SIZE = 1024
        
        # STT objects
        self.stt_audio = pyaudio.PyAudio()
        self.speech_client = None
        self.init_google_client()
        
        # STT buffers
        self.audio_buffer = []
        self.silence_buffer = []
        self.current_phrase = ""
        self.silence_start = None
        self.max_silence_duration = 2.0
        
        # STT settings
        self.BUFFER_DURATION = 3.0
        self.BUFFER_SIZE_STT = int(self.STT_SAMPLE_RATE * self.BUFFER_DURATION)
        self.SILENCE_DURATION = 1.0
        self.SILENCE_THRESHOLD = 0.01
        
        # Text processing
        self.text_processor_thread = None
        
    def init_google_client(self):
        """Initialize Google Cloud Speech client"""
        try:
            credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
            if credentials_path:
                self.speech_client = speech.SpeechClient()
                print("✅ STT: Google Cloud Speech client initialized")
            else:
                print("⚠️  STT: Google Cloud credentials not found")
        except Exception as e:
            print(f"❌ STT: Failed to initialize Google Cloud client: {e}")
    
    def transcribe_audio(self, audio_chunk):
        """Transcribe audio chunk using Google Speech-to-Text - optimized for continuous speech"""
        if not self.speech_client:
            return None
        
        try:
            print("🔍 STT: Transcribing continuous speech...")
            
            audio = speech.RecognitionAudio(content=audio_chunk)
            config = speech.RecognitionConfig(
                encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
                sample_rate_hertz=self.STT_SAMPLE_RATE,
                language_code="ru-RU",  # Russian
                enable_automatic_punctuation=True,
                model="latest_long",
                # Optimized settings for continuous speech
                enable_word_time_offsets=False,  # Disable for speed
                enable_word_confidence=False,    # Disable for speed
                use_enhanced=True,               # Use enhanced model for better accuracy
                max_alternatives=1               # Only get best result
            )
            
            response = self.speech_client.recognize(config=config, audio=audio)
            
            transcribed_text = ""
            for result in response.results:
                transcribed_text += result.alternatives[0].transcript + " "
            
            text = transcribed_text.strip()
            if text:
                print(f"📝 STT: Complete phrase transcribed: '{text}'")
                return text
            else:
                print("🔇 STT: No speech detected in phrase")
                return None
            
        except Exception as e:
            print(f"❌ STT: Transcription error: {e}")
            return None
    

    
    def add_text_to_queue(self, phrase):
        """Add transcribed text to in-memory queue for TTS processing"""
        try:
            # Add to queue (non-blocking)
            self.text_queue.put_nowait(phrase)
            print(f"📝 STT → TTS Queue: '{phrase}'")
            
            # Log to file for testing
            self.log_to_file(f"ADDED: {phrase}")
            
        except queue.Full:
            # Queue is full, remove oldest item and add new one (LIFO behavior)
            try:
                old_phrase = self.text_queue.get_nowait()  # Remove oldest
                self.text_queue.put_nowait(phrase)  # Add newest
                print(f"🔄 STT → TTS Queue: Replaced '{old_phrase}' with '{phrase}'")
                
                # Log to file for testing
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
        """STT worker thread - uses proper streaming recognition for continuous speech"""
        print("🎤 STT Worker: Starting continuous streaming recognition...")
        
        try:
            stream = self.stt_audio.open(
                format=self.STT_FORMAT,
                channels=self.STT_CHANNELS,
                rate=self.STT_SAMPLE_RATE,
                input=True,
                frames_per_buffer=self.STT_CHUNK_SIZE
            )
            
            print("✅ STT: Audio input stream started")
            
            # Use proper streaming recognition for continuous speech
            # Buffer for accumulating audio until natural silence
            audio_buffer = b""
            silence_start = None
            silence_threshold = 0.01
            min_phrase_duration = 2.0  # Minimum 2 seconds before processing
            max_phrase_duration = 10.0  # Maximum 10 seconds per phrase
            silence_duration = 1.5  # 1.5 seconds of silence to end phrase
            
            phrase_start_time = time.time()
            
            while self.running:
                try:
                    # Read audio chunk
                    data = stream.read(self.STT_CHUNK_SIZE, exception_on_overflow=False)
                    audio_buffer += data
                    
                    # Check audio level for silence detection
                    audio_data = np.frombuffer(data, dtype=np.int16)
                    audio_level = np.sqrt(np.mean(audio_data.astype(np.float32) ** 2)) / 32768.0
                    
                    current_time = time.time()
                    phrase_duration = current_time - phrase_start_time
                    
                    # Check for silence
                    if audio_level < silence_threshold:
                        if silence_start is None:
                            silence_start = current_time
                    else:
                        silence_start = None
                    
                    # Determine when to process the phrase
                    should_process = False
                    
                    # Process if we have enough audio and silence detected
                    if (len(audio_buffer) > 0 and 
                        phrase_duration >= min_phrase_duration and
                        silence_start is not None and
                        (current_time - silence_start) >= silence_duration):
                        should_process = True
                    
                    # Also process if we've reached max duration (force processing)
                    elif phrase_duration >= max_phrase_duration and len(audio_buffer) > 0:
                        should_process = True
                    
                    if should_process:
                        print(f"🎤 STT: Processing {len(audio_buffer)} bytes ({phrase_duration:.1f}s) - silence detected")
                        
                        # Transcribe the complete phrase
                        transcribed_text = self.transcribe_audio(audio_buffer)
                        
                        if transcribed_text:
                            print(f"📝 STT: Complete phrase: '{transcribed_text}'")
                            
                            # Add to queue for TTS processing
                            self.add_text_to_queue(transcribed_text)
                        else:
                            print("🔇 STT: No speech detected in phrase")
                        
                        # Reset for next phrase
                        audio_buffer = b""
                        silence_start = None
                        phrase_start_time = current_time
                    
                    time.sleep(0.01)  # Small delay
                    
                except Exception as e:
                    print(f"❌ STT: Error in audio processing: {e}")
                    break
            
            # Cleanup
            stream.stop_stream()
            stream.close()
            self.stt_audio.terminate()
            print("✅ STT Worker stopped")
            
        except Exception as e:
            print(f"❌ STT Worker error: {e}")
    
    def text_processor_worker(self):
        """Process text from in-memory queue and trigger TTS"""
        print("📝 Text Processor: Monitoring in-memory queue...")
        
        while self.running:
            try:
                # Get text from queue (blocking with timeout)
                try:
                    text = self.text_queue.get(timeout=0.1)
                    print(f"🎵 TTS: Processing text from queue: '{text}'")
                    
                    # Log processing to file
                    self.log_to_file(f"PROCESSING: {text}")
                    
                    # Start TTS streaming for this text
                    asyncio.run(self.stream_text_to_tts(text))
                    
                    # Mark task as done
                    self.text_queue.task_done()
                    
                except queue.Empty:
                    # No text in queue, continue monitoring
                    continue
                    
            except Exception as e:
                print(f"❌ Text processor error: {e}")
                time.sleep(0.1)
    
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
    
    async def stream_text_to_tts(self, text):
        """Stream text to TTS with smooth live playback"""
        if not text.strip():
            return
        
        # Split text into chunks
        text_chunks = self._split_text_into_chunks(text, max_chunk_size=100)
        
        # Connect to ElevenLabs
        import websockets
        uri = "wss://api.elevenlabs.io/v1/text-to-speech/GN4wbsbejSnGSa1AzjH5/stream-input?model_id=eleven_multilingual_v2&optimize_streaming_latency=4"
        
        try:
            async with websockets.connect(uri) as websocket:
                # Get API key
                api_key = self._get_api_key()
                if not api_key:
                    print("❌ API key not found. Skipping TTS.")
                    return
                
                # Send initialization
                init_message = {
                    "text": " ",
                    "voice_settings": {
                        "stability": 0.5,
                        "similarity_boost": 0.75
                    },
                    "xi_api_key": api_key
                }
                await websocket.send(json.dumps(init_message))
                
                # Send text chunks
                for chunk in text_chunks:
                    message = {
                        "text": chunk,
                        "xi_api_key": api_key
                    }
                    await websocket.send(json.dumps(message))
                
                # Send end marker
                end_message = {
                    "text": "",
                    "xi_api_key": api_key
                }
                await websocket.send(json.dumps(end_message))
                
                # Start smooth audio streaming
                await self._smooth_audio_streaming(websocket)
                
        except Exception as e:
            print(f"❌ TTS streaming error: {e}")
    
    def _split_text_into_chunks(self, text, max_chunk_size=100):
        """Split text into chunks while preserving spacing"""
        chunks = []
        current_chunk = ""
        current_length = 0
        
        import re
        parts = re.split(r'(\s+)', text)
        
        for part in parts:
            if current_length + len(part) > max_chunk_size and current_chunk:
                chunks.append(current_chunk)
                current_chunk = part
                current_length = len(part)
            else:
                current_chunk += part
                current_length += len(part)
        
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks
    
    async def _smooth_audio_streaming(self, websocket):
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
            
            # Track metrics
            total_audio_chunks = 0
            total_audio_bytes = 0
            
            # Receive and process audio chunks
            async for message in websocket:
                try:
                    data = json.loads(message)
                    
                    if "audio" in data and data["audio"]:
                        # Decode audio
                        audio_data = base64.b64decode(data["audio"])
                        total_audio_chunks += 1
                        total_audio_bytes += len(audio_data)
                        
                        # Add to continuous buffer (non-blocking)
                        try:
                            self.continuous_buffer.put_nowait(audio_data)
                        except queue.Full:
                            # Buffer full, skip this chunk to maintain real-time
                            pass
                        
                        # Collect for file saving
                        self.mp3_chunks.append(audio_data)
                        self.all_audio_chunks.append(audio_data)  # Save all audio for final file
                        
                        # Start playback when we have enough data
                        if not self.playback_started and total_audio_chunks >= self.BUFFER_SIZE:
                            self.playback_started = True
                            print(f"🎵 Starting continuous playback (buffer: {total_audio_chunks} chunks)")
                    
                    elif "audio" in data and data["audio"] is None:
                        print("📡 End of stream signal received")
                    
                    if data.get("isFinal"):
                        print("✅ TTS stream completed")
                        break
                        
                except json.JSONDecodeError as e:
                    print(f"⚠️ JSON decode error: {e}")
                    continue
                except Exception as e:
                    print(f"❌ Message processing error: {e}")
                    break
            
            # Don't close stream here - keep it open for continuous playback
            
        except Exception as e:
            print(f"❌ Smooth streaming error: {e}")
    
    def _continuous_playback_worker(self, stream):
        """Continuous playback worker that eliminates gaps"""
        try:
            while self.running:
                try:
                    # Get audio chunk (blocking with timeout)
                    audio_data = self.continuous_buffer.get(timeout=0.1)
                    
                    # Decode and play immediately
                    self._play_audio_chunk_smooth(audio_data, stream)
                    
                except queue.Empty:
                    # No audio data, continue waiting
                    continue
                except Exception as e:
                    print(f"⚠️ Playback error: {e}")
                    continue
            
            print("🎵 Continuous playback worker completed")
            
        except Exception as e:
            print(f"❌ Continuous playback worker error: {e}")
    
    def _play_audio_chunk_smooth(self, mp3_data, stream):
        """Play audio chunk with minimal processing for smooth playback"""
        try:
            if len(mp3_data) < 100:
                return
            
            try:
                audio_segment = AudioSegment.from_file(io.BytesIO(mp3_data), format="mp3")
            except Exception:
                return
            
            if len(audio_segment) == 0:
                return
            
            pcm_data = audio_segment.get_array_of_samples()
            pcm_float = np.array(pcm_data, dtype=np.float32) / 32768.0
            
            # Play immediately without additional delays
            stream.write(pcm_float.astype(np.float32).tobytes())
            
        except Exception as e:
            # Silent error handling to avoid gaps
            pass
    
    def start(self):
        """Start the real-time voice changer"""
        print("🎤🎵 Real-time Voice Changer with Smooth Live Playback")
        print("=" * 60)
        print("🎤 STT: Captures your speech → adds to queue")
        print("📝 Text Processor: Processes from memory queue")
        print("🎵 TTS: Streams text → smooth live playback")
        print("📁 Queue log: queue_log.txt")
        print("🎵 Output file: realtime_voice_changer_output.mp3")
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
        
        print("✅ Real-time voice changer started!")
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
    voice_changer = RealtimeVoiceChangerSmooth()
    voice_changer.start()

if __name__ == "__main__":
    asyncio.run(main()) 