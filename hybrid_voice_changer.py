#!/usr/bin/env python3
"""
Hybrid Real-time Voice Changer
Uses WebSocket for speed, falls back to REST API for reliability
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
import requests

# Load environment variables
load_dotenv()

class HybridVoiceChanger:
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
        
        # Hybrid streaming settings
        self.BUFFER_DURATION = 2.0
        self.BUFFER_SIZE = int(self.SAMPLE_RATE * self.BUFFER_DURATION)
        self.OVERLAP_DURATION = 0.5
        self.OVERLAP_SIZE = int(self.SAMPLE_RATE * self.OVERLAP_DURATION)
        self.PROCESSING_INTERVAL = 0.3
        
        # Connection settings
        self.MAX_QUEUE_SIZE = 5
        self.WEBSOCKET_TIMEOUT = 5.0
        self.REST_TIMEOUT = 15.0
        self.MAX_WEBSOCKET_FAILURES = 2
        
        # Queues
        self.audio_queue = queue.Queue(maxsize=self.MAX_QUEUE_SIZE)
        self.tts_queue = queue.Queue(maxsize=self.MAX_QUEUE_SIZE)
        self.playback_queue = queue.Queue(maxsize=self.MAX_QUEUE_SIZE)
        
        # Control flags
        self.is_running = False
        self.is_speaking = False
        
        # Latency tracking
        self.latency_stats = {
            'stt_processing': [],
            'tts_websocket': [],
            'tts_rest': [],
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
        
        # Connection management
        self.tts_websocket = None
        self.websocket_failures = 0
        self.use_websocket = True
        self.last_connection_attempt = 0
        self.connection_cooldown = 1.0
        
        # Initialize Google client
        self.speech_client = None
        if self.GOOGLE_APPLICATION_CREDENTIALS:
            self.speech_client = speech.SpeechClient()
    
    def start_streaming(self):
        """Start the hybrid voice changer"""
        if not self.ELEVENLABS_API_KEY:
            print("❌ ELEVENLABS_API_KEY not found")
            return False
        
        if not self.GOOGLE_APPLICATION_CREDENTIALS:
            print("❌ GOOGLE_APPLICATION_CREDENTIALS not found")
            return False
        
        print("🔄 Starting Hybrid Voice Changer...")
        print("   WebSocket for speed, REST API for reliability")
        print("   Automatic fallback on connection issues")
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
            print("\n🛑 Stopping hybrid voice changer...")
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
        print("✅ Hybrid voice changer stopped")
    
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
        print("📊 HYBRID VOICE CHANGER PERFORMANCE REPORT")
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
            tts_websocket_avg = statistics.mean(self.latency_stats['tts_websocket']) if self.latency_stats['tts_websocket'] else 0
            tts_rest_avg = statistics.mean(self.latency_stats['tts_rest']) if self.latency_stats['tts_rest'] else 0
            
            print(f"\n⚡ PIPELINE EFFICIENCY:")
            print(f"Total Pipeline Latency: {total_avg:.3f}s")
            print(f"STT Processing: {stt_avg:.3f}s ({(stt_avg/total_avg)*100:.1f}%)")
            print(f"TTS WebSocket: {tts_websocket_avg:.3f}s ({(tts_websocket_avg/total_avg)*100:.1f}%)")
            print(f"TTS REST API: {tts_rest_avg:.3f}s ({(tts_rest_avg/total_avg)*100:.1f}%)")
            
            print(f"\n🔄 HYBRID METRICS:")
            print(f"WebSocket Failures: {self.websocket_failures}")
            print(f"Current Mode: {'WebSocket' if self.use_websocket else 'REST API'}")
            print(f"Max Queue Size: {self.MAX_QUEUE_SIZE}")
        
        print("="*60)
    
    def _audio_recording_thread(self):
        """Hybrid audio recording"""
        try:
            stream = self.audio.open(
                format=self.FORMAT,
                channels=self.CHANNELS,
                rate=self.SAMPLE_RATE,
                input=True,
                frames_per_buffer=self.CHUNK_SIZE
            )
            
            print("🎤 Hybrid recording started...")
            
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
        """Hybrid STT processing"""
        print("🔍 Hybrid STT processing started...")
        
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
        """Hybrid TTS processing with fallback"""
        print("🎵 Hybrid TTS processing started...")
        
        while self.is_running:
            try:
                # Non-blocking queue get with timeout
                try:
                    text_data = self.tts_queue.get(timeout=1.0)
                except queue.Empty:
                    continue
                
                text, pipeline_start_time = text_data
                tts_start_time = time.time()
                
                # Try WebSocket first, fallback to REST
                success = False
                if self.use_websocket:
                    success = asyncio.run(self._tts_websocket(text, pipeline_start_time, tts_start_time))
                
                if not success:
                    print("🔄 Falling back to REST API...")
                    success = self._tts_rest_api(text, pipeline_start_time, tts_start_time)
                
                if not success:
                    print("⚠️ TTS failed completely, skipping text")
                
            except Exception as e:
                print(f"❌ TTS processing error: {e}")
    
    async def _tts_websocket(self, text, pipeline_start_time, tts_start_time):
        """TTS using WebSocket"""
        try:
            # Check connection cooldown
            current_time = time.time()
            if current_time - self.last_connection_attempt < self.connection_cooldown:
                return False
            
            # Establish connection if needed
            if not await self._ensure_websocket_connection():
                return False
            
            # Send text
            text_message = {
                "text": text,
                "try_trigger_generation": True
            }
            await self.tts_websocket.send(json.dumps(text_message))
            
            # Send end marker
            await self.tts_websocket.send(json.dumps({"text": ""}))
            
            # Receive audio
            success = await self._receive_websocket_audio(pipeline_start_time, tts_start_time)
            
            if success:
                return True
            else:
                # Mark WebSocket failure
                self.websocket_failures += 1
                if self.websocket_failures >= self.MAX_WEBSOCKET_FAILURES:
                    print("🔄 Switching to REST API mode due to WebSocket failures")
                    self.use_websocket = False
                return False
            
        except Exception as e:
            print(f"❌ WebSocket TTS error: {e}")
            self.websocket_failures += 1
            await self._close_tts_connection()
            self.tts_websocket = None
            return False
    
    async def _ensure_websocket_connection(self):
        """Ensure WebSocket connection is established"""
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
                timeout=self.WEBSOCKET_TIMEOUT
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
            
            print("✅ WebSocket connection established")
            return True
            
        except Exception as e:
            print(f"❌ WebSocket connection failed: {e}")
            return False
    
    async def _receive_websocket_audio(self, pipeline_start_time, tts_start_time):
        """Receive audio from WebSocket"""
        try:
            self.is_speaking = True
            first_audio_received = False
            
            async with asyncio.timeout(10.0):
                async for message in self.tts_websocket:
                    try:
                        data = json.loads(message)
                        
                        if "audio" in data and data["audio"]:
                            if not first_audio_received:
                                tts_first_audio_time = time.time()
                                tts_latency = tts_first_audio_time - tts_start_time
                                self.latency_stats['tts_websocket'].append(tts_latency)
                                first_audio_received = True
                            
                            # Process audio data
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
                        print(f"⚠️ WebSocket JSON decode error: {e}")
                        continue
                    except Exception as e:
                        print(f"❌ WebSocket message error: {e}")
                        break
            
            self.is_speaking = False
            return first_audio_received
            
        except asyncio.TimeoutError:
            print("⚠️ WebSocket timeout")
            self.is_speaking = False
            return False
        except Exception as e:
            print(f"❌ WebSocket audio reception error: {e}")
            self.is_speaking = False
            return False
    
    def _tts_rest_api(self, text, pipeline_start_time, tts_start_time):
        """TTS using REST API (fallback)"""
        try:
            print("🌐 Using REST API for TTS...")
            
            url = f"https://api.elevenlabs.io/v1/text-to-speech/{self.VOICE_ID}"
            
            headers = {
                "Accept": "audio/mpeg",
                "Content-Type": "application/json",
                "xi-api-key": self.ELEVENLABS_API_KEY
            }
            
            data = {
                "text": text,
                "model_id": self.MODEL_ID,
                "voice_settings": {
                    "stability": 0.5,
                    "similarity_boost": 0.8
                }
            }
            
            # Make REST API request
            response = requests.post(url, json=data, headers=headers, timeout=self.REST_TIMEOUT)
            
            if response.status_code == 200:
                # Track REST API latency
                tts_end_time = time.time()
                tts_latency = tts_end_time - tts_start_time
                self.latency_stats['tts_rest'].append(tts_latency)
                
                # Add to playback queue
                try:
                    self.playback_queue.put_nowait((response.content, pipeline_start_time))
                    print(f"✅ REST API TTS successful (latency: {tts_latency:.3f}s)")
                    return True
                except queue.Full:
                    print("⚠️ Playback queue full, skipping audio")
                    return False
            else:
                print(f"❌ REST API error: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            print(f"❌ REST API TTS error: {e}")
            return False
    
    def _audio_playback_thread(self):
        """Hybrid audio playback"""
        print("🔊 Hybrid audio playback started...")
        
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
    voice_changer = HybridVoiceChanger()
    voice_changer.start_streaming()

if __name__ == "__main__":
    main() 