#!/usr/bin/env python3
"""
Robust Real-time Voice Changer
Handles ElevenLabs connection issues and prevents latency degradation
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
import threading
import queue
import time
from collections import deque
import statistics

# Load environment variables
load_dotenv()

class RobustVoiceChanger:
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
        
        # Robust streaming settings
        self.BUFFER_DURATION = 2.0
        self.BUFFER_SIZE = int(self.SAMPLE_RATE * self.BUFFER_DURATION)
        self.OVERLAP_DURATION = 0.5
        self.OVERLAP_SIZE = int(self.SAMPLE_RATE * self.OVERLAP_DURATION)
        self.PROCESSING_INTERVAL = 0.3
        
        # Robust connection settings
        self.MAX_QUEUE_SIZE = 5  # Smaller queues
        self.MAX_RETRIES = 3
        self.CONNECTION_TIMEOUT = 5.0
        self.MESSAGE_TIMEOUT = 10.0
        
        # Queues with size limits
        self.audio_queue = queue.Queue(maxsize=self.MAX_QUEUE_SIZE)
        self.tts_queue = queue.Queue(maxsize=self.MAX_QUEUE_SIZE)
        self.playback_queue = queue.Queue(maxsize=self.MAX_QUEUE_SIZE)
        
        # Control flags
        self.is_running = False
        self.is_speaking = False
        
        # Latency tracking
        self.latency_stats = {
            'stt_processing': [],
            'tts_processing': [],
            'total_pipeline': []
        }
        self.session_start_time = None
        self.total_phrases_processed = 0
        
        # Audio buffer
        self.audio_buffer = deque(maxlen=int(self.SAMPLE_RATE * 5))
        self.last_processed_position = 0
        self.last_processing_time = 0
        
        # PyAudio instance
        self.audio = pyaudio.PyAudio()
        
        # Robust TTS connection management
        self.tts_websocket = None
        self.tts_connection_time = 0
        self.connection_errors = 0
        self.last_connection_attempt = 0
        self.connection_cooldown = 2.0  # Wait 2s between connection attempts
        
        # Initialize Google client
        self.speech_client = None
        if self.GOOGLE_APPLICATION_CREDENTIALS:
            self.speech_client = speech.SpeechClient()
    
    def start_streaming(self):
        """Start the robust voice changer"""
        if not self.ELEVENLABS_API_KEY:
            print("❌ ELEVENLABS_API_KEY not found")
            return False
        
        if not self.GOOGLE_APPLICATION_CREDENTIALS:
            print("❌ GOOGLE_APPLICATION_CREDENTIALS not found")
            return False
        
        print("🛡️ Starting Robust Voice Changer...")
        print("   Handles connection issues automatically")
        print("   Prevents latency degradation")
        print("   Press Ctrl+C to stop")
        
        self.session_start_time = time.time()
        self.is_running = True
        
        # Start threads
        recording_thread = threading.Thread(target=self._audio_recording_thread)
        stt_thread = threading.Thread(target=self._stt_processing_thread)
        tts_thread = threading.Thread(target=self._tts_processing_thread)
        playback_thread = threading.Thread(target=self._audio_playback_thread)
        
        recording_thread.start()
        stt_thread.start()
        tts_thread.start()
        playback_thread.start()
        
        try:
            while self.is_running:
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("\n🛑 Stopping robust voice changer...")
            self.is_running = False
            self._print_latency_report()
        
        # Cleanup
        if self.tts_websocket:
            asyncio.run(self._close_tts_connection())
        
        recording_thread.join()
        stt_thread.join()
        tts_thread.join()
        playback_thread.join()
        
        self.audio.terminate()
        print("✅ Robust voice changer stopped")
    
    async def _close_tts_connection(self):
        """Close TTS WebSocket connection"""
        if self.tts_websocket:
            try:
                await self.tts_websocket.close()
            except:
                pass
            self.tts_websocket = None
    
    def _print_latency_report(self):
        """Print comprehensive latency report"""
        if not self.latency_stats['total_pipeline']:
            print("📊 No latency data collected")
            return
        
        print("\n" + "="*60)
        print("📊 ROBUST VOICE CHANGER PERFORMANCE REPORT")
        print("="*60)
        
        session_duration = time.time() - self.session_start_time
        print(f"Session Duration: {session_duration:.1f}s")
        print(f"Total Phrases Processed: {self.total_phrases_processed}")
        print(f"Average Phrases per Minute: {self.total_phrases_processed / (session_duration / 60):.1f}")
        
        print("\n🔍 LATENCY BREAKDOWN:")
        print("-" * 40)
        
        for metric, values in self.latency_stats.items():
            if values:
                avg = statistics.mean(values)
                median = statistics.median(values)
                min_val = min(values)
                max_val = max(values)
                std_dev = statistics.stdev(values) if len(values) > 1 else 0
                
                print(f"{metric.replace('_', ' ').title()}:")
                print(f"  Avg: {avg:.3f}s | Median: {median:.3f}s | Min: {min_val:.3f}s | Max: {max_val:.3f}s | Std: {std_dev:.3f}s")
                print(f"  Samples: {len(values)}")
            else:
                print(f"{metric.replace('_', ' ').title()}: No data")
        
        # Calculate pipeline efficiency
        if self.latency_stats['total_pipeline']:
            total_avg = statistics.mean(self.latency_stats['total_pipeline'])
            stt_avg = statistics.mean(self.latency_stats['stt_processing']) if self.latency_stats['stt_processing'] else 0
            tts_avg = statistics.mean(self.latency_stats['tts_processing']) if self.latency_stats['tts_processing'] else 0
            
            print(f"\n⚡ PIPELINE EFFICIENCY:")
            print(f"Total Pipeline Latency: {total_avg:.3f}s")
            print(f"STT Processing: {stt_avg:.3f}s ({(stt_avg/total_avg)*100:.1f}%)")
            print(f"TTS Processing: {tts_avg:.3f}s ({(tts_avg/total_avg)*100:.1f}%)")
            
            print(f"\n🛡️ ROBUSTNESS METRICS:")
            print(f"Connection Errors: {self.connection_errors}")
            print(f"Max Queue Size: {self.MAX_QUEUE_SIZE}")
            print(f"Connection Timeout: {self.CONNECTION_TIMEOUT}s")
            print(f"Message Timeout: {self.MESSAGE_TIMEOUT}s")
        
        print("="*60)
    
    def _audio_recording_thread(self):
        """Robust audio recording"""
        try:
            stream = self.audio.open(
                format=self.FORMAT,
                channels=self.CHANNELS,
                rate=self.SAMPLE_RATE,
                input=True,
                frames_per_buffer=self.CHUNK_SIZE
            )
            
            print("🎤 Robust recording started...")
            
            while self.is_running:
                try:
                    data = stream.read(self.CHUNK_SIZE, exception_on_overflow=False)
                    audio_chunk = np.frombuffer(data, dtype=np.int16)
                    
                    # Add to buffer
                    self.audio_buffer.extend(audio_chunk)
                    
                    # Check if we have enough new audio to process
                    current_time = time.time()
                    buffer_length = len(self.audio_buffer)
                    
                    if (current_time - self.last_processing_time > self.PROCESSING_INTERVAL and 
                        buffer_length >= self.BUFFER_SIZE):
                        
                        # Extract buffer with overlap
                        start_pos = max(0, self.last_processed_position - self.OVERLAP_SIZE)
                        end_pos = buffer_length
                        
                        if end_pos - start_pos >= self.BUFFER_SIZE:
                            # Extract audio chunk
                            stt_chunk = list(self.audio_buffer)[start_pos:end_pos]
                            
                            # Update processing position
                            self.last_processed_position = end_pos - self.OVERLAP_SIZE
                            self.last_processing_time = current_time
                            
                            # Try to add to queue (non-blocking)
                            try:
                                stt_audio = np.array(stt_chunk, dtype=np.int16).tobytes()
                                self.audio_queue.put_nowait(stt_audio)
                                print(f"🔄 Processing {len(stt_chunk)/self.SAMPLE_RATE:.1f}s audio")
                            except queue.Full:
                                print("⚠️ Audio queue full, skipping chunk")
                        
                except Exception as e:
                    print(f"❌ Recording error: {e}")
                    break
            
            stream.stop_stream()
            stream.close()
            
        except Exception as e:
            print(f"❌ Failed to start recording: {e}")
    
    def _stt_processing_thread(self):
        """Robust STT processing"""
        print("🔍 Robust STT processing started...")
        
        while self.is_running:
            try:
                # Non-blocking queue get with timeout
                try:
                    audio_data = self.audio_queue.get(timeout=1.0)
                except queue.Empty:
                    continue
                
                stt_start_time = time.time()
                
                # Transcribe audio
                text = self._transcribe_audio(audio_data)
                
                stt_end_time = time.time()
                stt_latency = stt_end_time - stt_start_time
                self.latency_stats['stt_processing'].append(stt_latency)
                
                if text and text.strip():
                    print(f"📝 STT: '{text}' (latency: {stt_latency:.3f}s)")
                    
                    # Try to add to TTS queue (non-blocking)
                    try:
                        self.tts_queue.put_nowait((text, stt_start_time))
                    except queue.Full:
                        print("⚠️ TTS queue full, skipping text")
                
            except Exception as e:
                print(f"❌ STT processing error: {e}")
    
    def _transcribe_audio(self, audio_data):
        """Transcribe audio using Google Speech-to-Text"""
        if not self.speech_client:
            return None
        
        try:
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
            print(f"❌ Transcription error: {e}")
            return None
    
    def _tts_processing_thread(self):
        """Robust TTS processing with connection management"""
        print("🎵 Robust TTS processing started...")
        
        while self.is_running:
            try:
                # Non-blocking queue get with timeout
                try:
                    text_data = self.tts_queue.get(timeout=1.0)
                except queue.Empty:
                    continue
                
                text, pipeline_start_time = text_data
                tts_start_time = time.time()
                
                # Convert to speech with retry logic
                success = asyncio.run(self._text_to_speech_stream_robust(text, pipeline_start_time, tts_start_time))
                
                if not success:
                    print("⚠️ TTS failed, skipping text")
                
            except Exception as e:
                print(f"❌ TTS processing error: {e}")
                self.connection_errors += 1
    
    async def _text_to_speech_stream_robust(self, text, pipeline_start_time, tts_start_time):
        """Robust TTS streaming with connection management"""
        if not self.ELEVENLABS_API_KEY:
            return False
        
        # Check connection cooldown
        current_time = time.time()
        if current_time - self.last_connection_attempt < self.connection_cooldown:
            return False
        
        for attempt in range(self.MAX_RETRIES):
            try:
                # Establish connection if needed
                if not await self._ensure_tts_connection():
                    print(f"⚠️ TTS connection failed (attempt {attempt + 1}/{self.MAX_RETRIES})")
                    continue
                
                # Send text
                text_message = {
                    "text": text,
                    "try_trigger_generation": True
                }
                await self.tts_websocket.send(json.dumps(text_message))
                
                # Send end marker
                await self.tts_websocket.send(json.dumps({"text": ""}))
                
                # Receive audio with timeout
                success = await self._receive_audio_with_timeout(pipeline_start_time, tts_start_time)
                
                if success:
                    return True
                else:
                    print(f"⚠️ TTS audio reception failed (attempt {attempt + 1}/{self.MAX_RETRIES})")
                    # Reset connection for next attempt
                    await self._close_tts_connection()
                    self.tts_websocket = None
                
            except Exception as e:
                print(f"❌ TTS attempt {attempt + 1} failed: {e}")
                self.connection_errors += 1
                await self._close_tts_connection()
                self.tts_websocket = None
        
        return False
    
    async def _ensure_tts_connection(self):
        """Ensure TTS connection is established"""
        if self.tts_websocket is not None:
            return True
        
        try:
            uri = (
                f"wss://api.elevenlabs.io/v1/text-to-speech/{self.VOICE_ID}/stream-input"
                f"?model_id={self.MODEL_ID}&optimize_streaming_latency=4"
            )
            headers = {
                "xi-api-key": self.ELEVENLABS_API_KEY,
                "Content-Type": "application/json"
            }
            
            self.last_connection_attempt = time.time()
            
            self.tts_websocket = await asyncio.wait_for(
                websockets.connect(uri, extra_headers=headers),
                timeout=self.CONNECTION_TIMEOUT
            )
            
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
            await self.tts_websocket.send(json.dumps(init_message))
            
            print("✅ TTS connection established")
            return True
            
        except Exception as e:
            print(f"❌ TTS connection failed: {e}")
            self.connection_errors += 1
            return False
    
    async def _receive_audio_with_timeout(self, pipeline_start_time, tts_start_time):
        """Receive audio with timeout"""
        try:
            self.is_speaking = True
            first_audio_received = False
            
            # Set timeout for receiving messages
            async with asyncio.timeout(self.MESSAGE_TIMEOUT):
                async for message in self.tts_websocket:
                    try:
                        data = json.loads(message)
                        
                        if "audio" in data and data["audio"]:
                            if not first_audio_received:
                                tts_first_audio_time = time.time()
                                tts_latency = tts_first_audio_time - tts_start_time
                                self.latency_stats['tts_processing'].append(tts_latency)
                                first_audio_received = True
                            
                            # Validate and process audio data
                            try:
                                audio_data = base64.b64decode(data["audio"])
                                if len(audio_data) > 0:
                                    try:
                                        self.playback_queue.put_nowait((audio_data, pipeline_start_time))
                                    except queue.Full:
                                        print("⚠️ Playback queue full, skipping audio")
                                else:
                                    print("⚠️ Received empty audio data")
                            except Exception as e:
                                print(f"⚠️ Audio data decode error: {e}")
                        
                        if data.get("isFinal"):
                            break
                            
                    except json.JSONDecodeError as e:
                        print(f"⚠️ TTS JSON decode error: {e}")
                        continue
                    except Exception as e:
                        print(f"❌ TTS message error: {e}")
                        break
            
            self.is_speaking = False
            return first_audio_received
            
        except asyncio.TimeoutError:
            print("⚠️ TTS message timeout")
            self.is_speaking = False
            return False
        except Exception as e:
            print(f"❌ TTS audio reception error: {e}")
            self.is_speaking = False
            return False
    
    def _audio_playback_thread(self):
        """Robust audio playback"""
        print("🔊 Robust audio playback started...")
        
        try:
            stream = self.audio.open(
                format=pyaudio.paFloat32,
                channels=1,
                rate=44100,
                output=True,
                frames_per_buffer=1024
            )
            
            while self.is_running:
                try:
                    # Non-blocking queue get with timeout
                    try:
                        audio_data = self.playback_queue.get(timeout=0.1)
                    except queue.Empty:
                        continue
                    
                    mp3_data, pipeline_start_time = audio_data
                    
                    # Track total pipeline latency
                    playback_start_time = time.time()
                    total_latency = playback_start_time - pipeline_start_time
                    self.latency_stats['total_pipeline'].append(total_latency)
                    self.total_phrases_processed += 1
                    
                    print(f"🔊 Playing (latency: {total_latency:.3f}s)")
                    
                    # Decode and play with error handling
                    from pydub import AudioSegment
                    import io
                    
                    try:
                        # Validate MP3 data
                        if len(mp3_data) < 100:
                            print(f"⚠️ Skipping invalid MP3 data (size: {len(mp3_data)} bytes)")
                            continue
                        
                        # Try to decode MP3
                        audio_segment = AudioSegment.from_file(io.BytesIO(mp3_data), format="mp3")
                        
                        # Check if audio segment is valid
                        if len(audio_segment) == 0:
                            print("⚠️ Skipping empty audio segment")
                            continue
                        
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
                        print(f"⚠️ Audio decoding error: {e}")
                        continue
                        
                except Exception as e:
                    print(f"❌ Playback error: {e}")
                    break
            
            stream.stop_stream()
            stream.close()
            
        except Exception as e:
            print(f"❌ Failed to start playback: {e}")

def main():
    """Main function"""
    voice_changer = RobustVoiceChanger()
    voice_changer.start_streaming()

if __name__ == "__main__":
    main() 