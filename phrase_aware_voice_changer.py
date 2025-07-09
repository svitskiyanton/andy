#!/usr/bin/env python3
"""
Phrase-Aware Voice Changer
Waits for natural sentence boundaries instead of cutting phrases
"""

import os
import asyncio
import json
import base64
import websockets
import pyaudio
import numpy as np
from dotenv import load_dotenv
from google.cloud import speech
import time
from collections import deque
import re

# Load environment variables
load_dotenv()

class PhraseAwareVoiceChanger:
    def __init__(self):
        self.ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
        self.VOICE_ID = os.getenv("VOICE_ID", "21m00Tcm4TlvDq8ikWAM")
        self.MODEL_ID = "eleven_multilingual_v2"
        self.GOOGLE_APPLICATION_CREDENTIALS = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
        
        # Audio settings
        self.SAMPLE_RATE = 16000
        self.CHUNK_SIZE = 1024
        self.CHANNELS = 1
        self.FORMAT = pyaudio.paInt16
        
        # Phrase-aware settings
        self.MIN_PHRASE_DURATION = 1.0  # Minimum 1 second
        self.MAX_PHRASE_DURATION = 8.0  # Maximum 8 seconds
        self.SILENCE_DURATION = 0.8  # 0.8 seconds of silence to end phrase
        self.SILENCE_THRESHOLD = 0.01
        
        # Control flags
        self.is_running = False
        
        # Latency tracking
        self.latency_stats = {
            'stt_processing': [],
            'tts_processing': [],
            'total_pipeline': []
        }
        self.session_start_time = None
        self.total_phrases_processed = 0
        
        # Audio buffer
        self.audio_buffer = deque(maxlen=int(self.SAMPLE_RATE * 15))
        self.silence_buffer = deque(maxlen=int(self.SAMPLE_RATE * 3))
        self.last_processing_time = 0
        
        # PyAudio instance
        self.audio = pyaudio.PyAudio()
        
        # Initialize Google client
        self.speech_client = None
        if self.GOOGLE_APPLICATION_CREDENTIALS:
            self.speech_client = speech.SpeechClient()
    
    def start_streaming(self):
        """Start the phrase-aware voice changer"""
        if not self.ELEVENLABS_API_KEY:
            print("❌ ELEVENLABS_API_KEY not found")
            return False
        
        if not self.GOOGLE_APPLICATION_CREDENTIALS:
            print("❌ GOOGLE_APPLICATION_CREDENTIALS not found")
            return False
        
        print("🎯 Starting Phrase-Aware Voice Changer...")
        print("   Waits for natural sentence boundaries")
        print("   No more cut-off phrases")
        print("   Press Ctrl+C to stop")
        
        self.session_start_time = time.time()
        self.is_running = True
        
        try:
            # Run the async main loop
            asyncio.run(self._main_loop())
        except KeyboardInterrupt:
            print("\n🛑 Stopping phrase-aware voice changer...")
            self.is_running = False
            self._print_latency_report()
        
        self.audio.terminate()
        print("✅ Phrase-aware voice changer stopped")
    
    def _print_latency_report(self):
        """Print comprehensive latency report"""
        if not self.latency_stats['total_pipeline']:
            print("📊 No latency data collected")
            return
        
        print("\n" + "="*60)
        print("📊 PHRASE-AWARE VOICE CHANGER PERFORMANCE REPORT")
        print("="*60)
        
        session_duration = time.time() - self.session_start_time
        print(f"Session Duration: {session_duration:.1f}s")
        print(f"Total Phrases Processed: {self.total_phrases_processed}")
        print(f"Average Phrases per Minute: {self.total_phrases_processed / (session_duration / 60):.1f}")
        
        print("\n🔍 LATENCY BREAKDOWN:")
        print("-" * 40)
        
        for metric, values in self.latency_stats.items():
            if values:
                avg = sum(values) / len(values)
                min_val = min(values)
                max_val = max(values)
                
                print(f"{metric.replace('_', ' ').title()}:")
                print(f"  Avg: {avg:.3f}s | Min: {min_val:.3f}s | Max: {max_val:.3f}s")
                print(f"  Samples: {len(values)}")
            else:
                print(f"{metric.replace('_', ' ').title()}: No data")
        
        print("="*60)
    
    async def _main_loop(self):
        """Main async loop with phrase-aware processing"""
        print("🎯 Starting phrase-aware processing loop...")
        
        # Start audio recording task
        recording_task = asyncio.create_task(self._phrase_aware_audio_recording())
        
        try:
            await recording_task
        except Exception as e:
            print(f"❌ Main loop error: {e}")
    
    async def _phrase_aware_audio_recording(self):
        """Phrase-aware audio recording with natural boundaries"""
        try:
            stream = self.audio.open(
                format=self.FORMAT,
                channels=self.CHANNELS,
                rate=self.SAMPLE_RATE,
                input=True,
                frames_per_buffer=self.CHUNK_SIZE
            )
            
            print("🎤 Phrase-aware audio recording started...")
            print(f"📊 Min phrase: {self.MIN_PHRASE_DURATION}s, Max phrase: {self.MAX_PHRASE_DURATION}s")
            print(f"📊 Silence threshold: {self.SILENCE_DURATION}s")
            
            while self.is_running:
                try:
                    # Read audio chunk
                    data = stream.read(self.CHUNK_SIZE, exception_on_overflow=False)
                    audio_chunk = np.frombuffer(data, dtype=np.int16)
                    
                    # Add to buffers
                    self.audio_buffer.extend(audio_chunk)
                    self.silence_buffer.extend(audio_chunk)
                    
                    # Check for phrase completion
                    current_time = time.time()
                    buffer_duration = len(self.audio_buffer) / self.SAMPLE_RATE
                    
                    # Check if we should process (silence + minimum duration)
                    if (current_time - self.last_processing_time > 0.5 and 
                        buffer_duration >= self.MIN_PHRASE_DURATION):
                        
                        # Check for silence
                        if len(self.silence_buffer) >= int(self.SAMPLE_RATE * self.SILENCE_DURATION):
                            recent_audio = list(self.silence_buffer)[-int(self.SAMPLE_RATE * self.SILENCE_DURATION):]
                            audio_level = np.sqrt(np.mean(np.array(recent_audio, dtype=np.float32) ** 2))
                            
                            # If silence detected, process the phrase
                            if audio_level < self.SILENCE_THRESHOLD * 32768:
                                # Extract complete phrase
                                phrase_audio = list(self.audio_buffer)
                                self.audio_buffer.clear()
                                self.silence_buffer.clear()
                                
                                phrase_duration = len(phrase_audio) / self.SAMPLE_RATE
                                print(f"🎯 Processing phrase ({phrase_duration:.1f}s)")
                                
                                # Process the complete phrase
                                await self._process_phrase(phrase_audio)
                                self.last_processing_time = current_time
                        
                        # Fallback: if phrase gets too long, process anyway
                        elif buffer_duration >= self.MAX_PHRASE_DURATION:
                            phrase_audio = list(self.audio_buffer)
                            self.audio_buffer.clear()
                            self.silence_buffer.clear()
                            
                            phrase_duration = len(phrase_audio) / self.SAMPLE_RATE
                            print(f"⏰ Max duration reached ({phrase_duration:.1f}s)")
                            
                            # Process the phrase
                            await self._process_phrase(phrase_audio)
                            self.last_processing_time = current_time
                    
                    # Small delay to prevent CPU overload
                    await asyncio.sleep(0.01)
                    
                except Exception as e:
                    print(f"❌ Recording error: {e}")
                    break
            
            stream.stop_stream()
            stream.close()
            
        except Exception as e:
            print(f"❌ Failed to start recording: {e}")
    
    async def _process_phrase(self, audio_chunk):
        """Process a complete phrase: STT → TTS → Playback"""
        try:
            pipeline_start = time.time()
            
            # Step 1: STT Processing
            stt_start = time.time()
            text = await self._transcribe_audio_async(audio_chunk)
            stt_end = time.time()
            stt_latency = stt_end - stt_start
            self.latency_stats['stt_processing'].append(stt_latency)
            
            if text and text.strip():
                # Clean up the text for better TTS
                cleaned_text = self._clean_text_for_tts(text)
                print(f"📝 STT: '{cleaned_text}' (latency: {stt_latency:.3f}s)")
                
                # Step 2: TTS Processing
                tts_start = time.time()
                success = await self._tts_streaming_async(cleaned_text, pipeline_start, tts_start)
                
                if success:
                    pipeline_end = time.time()
                    total_latency = pipeline_end - pipeline_start
                    self.latency_stats['total_pipeline'].append(total_latency)
                    self.total_phrases_processed += 1
                    print(f"🎯 Phrase complete (total latency: {total_latency:.3f}s)")
                else:
                    print("⚠️ TTS failed")
            else:
                print("🔇 No speech detected")
                
        except Exception as e:
            print(f"❌ Phrase processing error: {e}")
    
    def _clean_text_for_tts(self, text):
        """Clean text for better TTS output"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Ensure proper sentence ending
        if text and not text.endswith(('.', '!', '?', ':', ';')):
            text += '.'
        
        return text
    
    async def _transcribe_audio_async(self, audio_chunk):
        """Async transcription using Google STT"""
        if not self.speech_client:
            return None
        
        try:
            # Convert to bytes
            audio_data = np.array(audio_chunk, dtype=np.int16).tobytes()
            
            # Run STT in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            text = await loop.run_in_executor(None, self._transcribe_audio_sync, audio_data)
            return text
            
        except Exception as e:
            print(f"❌ Async transcription error: {e}")
            return None
    
    def _transcribe_audio_sync(self, audio_data):
        """Synchronous transcription (runs in thread pool)"""
        try:
            from google.cloud import speech
            
            audio = speech.RecognitionAudio(content=audio_data)
            config = speech.RecognitionConfig(
                encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
                sample_rate_hertz=self.SAMPLE_RATE,
                language_code="ru-RU",
                enable_automatic_punctuation=True,
                model="latest_long"
            )
            
            response = self.speech_client.recognize(config=config, audio=audio)
            
            transcribed_text = ""
            for result in response.results:
                transcribed_text += result.alternatives[0].transcript + " "
            
            return transcribed_text.strip()
            
        except Exception as e:
            print(f"❌ Sync transcription error: {e}")
            return None
    
    async def _tts_streaming_async(self, text, pipeline_start, tts_start):
        """Async TTS streaming with immediate playback"""
        try:
            print("🎵 Starting TTS streaming...")
            
            # WebSocket connection
            uri = (
                f"wss://api.elevenlabs.io/v1/text-to-speech/{self.VOICE_ID}/stream-input"
                f"?model_id={self.MODEL_ID}&optimize_streaming_latency=4"
            )
            headers = {
                "xi-api-key": self.ELEVENLABS_API_KEY,
                "Content-Type": "application/json"
            }
            
            async with websockets.connect(uri, extra_headers=headers) as websocket:
                print("✅ WebSocket connected")
                
                # Send initialization
                init_message = {
                    "text": " ",
                    "voice_settings": {
                        "speed": 1,
                        "stability": 0.5,
                        "similarity_boost": 0.8
                    },
                    "xi_api_key": self.ELEVENLABS_API_KEY
                }
                await websocket.send(json.dumps(init_message))
                
                # Send text
                text_message = {
                    "text": text,
                    "try_trigger_generation": True
                }
                await websocket.send(json.dumps(text_message))
                
                # Send end marker
                await websocket.send(json.dumps({"text": ""}))
                
                # Initialize audio playback
                stream = self.audio.open(
                    format=pyaudio.paFloat32,
                    channels=1,
                    rate=44100,
                    output=True,
                    frames_per_buffer=1024
                )
                
                print("🔊 Audio playback initialized")
                
                # Track TTS metrics
                first_audio_received = False
                audio_chunks_received = 0
                
                # Receive and play audio
                async for message in websocket:
                    try:
                        data = json.loads(message)
                        
                        if "audio" in data and data["audio"]:
                            # Track first audio timing
                            if not first_audio_received:
                                tts_first_audio_time = time.time()
                                tts_latency = tts_first_audio_time - tts_start
                                self.latency_stats['tts_processing'].append(tts_latency)
                                first_audio_received = True
                                print(f"🎵 First audio received (latency: {tts_latency:.3f}s)")
                            
                            # Decode and play audio immediately
                            audio_data = base64.b64decode(data["audio"])
                            audio_chunks_received += 1
                            
                            # Play audio chunk
                            await self._play_audio_chunk_async(audio_data, stream)
                            
                            print(f"🔊 Audio chunk {audio_chunks_received}: {len(audio_data)} bytes")
                        
                        if data.get("isFinal"):
                            print("✅ TTS stream completed")
                            break
                            
                    except json.JSONDecodeError as e:
                        print(f"⚠️ JSON decode error: {e}")
                        continue
                    except Exception as e:
                        print(f"❌ TTS message error: {e}")
                        break
                
                # Cleanup
                stream.stop_stream()
                stream.close()
                
                return first_audio_received
                
        except Exception as e:
            print(f"❌ TTS streaming error: {e}")
            return False
    
    async def _play_audio_chunk_async(self, mp3_data, stream):
        """Async audio chunk playback"""
        try:
            from pydub import AudioSegment
            import io
            
            # Validate MP3 data
            if len(mp3_data) < 100:
                return
            
            # Run audio processing in thread pool
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._play_audio_chunk_sync, mp3_data, stream)
            
        except Exception as e:
            print(f"⚠️ Async audio playback error: {e}")
    
    def _play_audio_chunk_sync(self, mp3_data, stream):
        """Synchronous audio chunk playback (runs in thread pool)"""
        try:
            from pydub import AudioSegment
            import io
            
            # Decode MP3 to PCM
            audio_segment = AudioSegment.from_file(io.BytesIO(mp3_data), format="mp3")
            
            if len(audio_segment) == 0:
                return
            
            # Convert to PCM
            pcm_data = audio_segment.get_array_of_samples()
            pcm_float = np.array(pcm_data, dtype=np.float32) / 32768.0
            
            # Play in chunks
            chunk_size = 1024
            for i in range(0, len(pcm_float), chunk_size):
                chunk = pcm_float[i:i + chunk_size]
                if len(chunk) < chunk_size:
                    chunk = np.pad(chunk, (0, chunk_size - len(chunk)))
                stream.write(chunk.astype(np.float32).tobytes())
                
        except Exception as e:
            print(f"⚠️ Sync audio playback error: {e}")

def main():
    """Main function"""
    voice_changer = PhraseAwareVoiceChanger()
    voice_changer.start_streaming()

if __name__ == "__main__":
    main() 