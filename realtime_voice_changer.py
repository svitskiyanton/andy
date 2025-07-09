#!/usr/bin/env python3
"""
Real-time Voice Changer
Streams audio through Google STT → ElevenLabs TTS pipeline in real-time
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

class RealtimeVoiceChanger:
    def __init__(self):
        self.ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
        self.VOICE_ID = os.getenv("VOICE_ID", "21m00Tcm4TlvDq8ikWAM")  # Rachel voice
        self.MODEL_ID = "eleven_multilingual_v2"
        self.GOOGLE_APPLICATION_CREDENTIALS = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
        
        # Audio settings
        self.SAMPLE_RATE = 16000
        self.CHUNK_SIZE = 1024
        self.CHANNELS = 1
        self.FORMAT = pyaudio.paInt16
        
        # Streaming settings
        self.STT_CHUNK_DURATION = 5.0  # seconds per STT chunk (increased for longer phrases)
        self.STT_CHUNK_SIZE = int(self.SAMPLE_RATE * self.STT_CHUNK_DURATION)
        self.SILENCE_THRESHOLD = 0.01
        self.SILENCE_DURATION = 1.5  # seconds of silence to trigger processing
        self.MIN_AUDIO_LENGTH = 1.0  # minimum seconds of audio before processing
        
        # Queues for communication between threads
        self.audio_queue = queue.Queue()
        self.tts_queue = queue.Queue()
        self.playback_queue = queue.Queue()
        
        # Control flags
        self.is_running = False
        self.is_speaking = False
        
        # Latency tracking
        self.latency_stats = {
            'stt_network': [],
            'stt_processing': [],
            'tts_network': [],
            'tts_processing': [],
            'audio_playback': [],
            'total_pipeline': []
        }
        self.session_start_time = None
        self.total_phrases_processed = 0
        
        # Initialize Google client
        self.speech_client = None
        if self.GOOGLE_APPLICATION_CREDENTIALS:
            self.speech_client = speech.SpeechClient()
        
        # Audio buffer for STT
        self.audio_buffer = deque(maxlen=int(self.SAMPLE_RATE * 15))  # 15 second buffer
        self.silence_buffer = deque(maxlen=int(self.SAMPLE_RATE * 3))  # 3 second silence detection
        self.last_processing_time = 0
        self.processing_cooldown = 0.5  # minimum seconds between processing
        
        # PyAudio instance
        self.audio = pyaudio.PyAudio()
        
    def start_streaming(self):
        """Start the real-time voice changer"""
        if not self.ELEVENLABS_API_KEY:
            print("❌ ELEVENLABS_API_KEY not found")
            return False
        
        if not self.GOOGLE_APPLICATION_CREDENTIALS:
            print("❌ GOOGLE_APPLICATION_CREDENTIALS not found")
            return False
        
        print("🎤 Starting Real-time Voice Changer...")
        print("   Speak in Russian for best results!")
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
            # Keep main thread alive
            while self.is_running:
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("\n🛑 Stopping voice changer...")
            self.is_running = False
            self._print_latency_report()
        
        # Wait for threads to finish
        recording_thread.join()
        stt_thread.join()
        tts_thread.join()
        playback_thread.join()
        
        self.audio.terminate()
        print("✅ Voice changer stopped")
    
    def _print_latency_report(self):
        """Print comprehensive latency report"""
        if not self.latency_stats['total_pipeline']:
            print("📊 No latency data collected")
            return
        
        print("\n" + "="*60)
        print("📊 LATENCY PERFORMANCE REPORT")
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
            print(f"Network Overhead: {total_avg - stt_avg - tts_avg:.3f}s")
        
        print("="*60)
    
    def _audio_recording_thread(self):
        """Thread for continuous audio recording with silence detection"""
        try:
            stream = self.audio.open(
                format=self.FORMAT,
                channels=self.CHANNELS,
                rate=self.SAMPLE_RATE,
                input=True,
                frames_per_buffer=self.CHUNK_SIZE
            )
            
            print("🎤 Recording started...")
            print("   Speak naturally - processing will trigger on silence")
            
            while self.is_running:
                try:
                    data = stream.read(self.CHUNK_SIZE, exception_on_overflow=False)
                    audio_chunk = np.frombuffer(data, dtype=np.int16)
                    
                    # Add to buffers
                    self.audio_buffer.extend(audio_chunk)
                    self.silence_buffer.extend(audio_chunk)
                    
                    # Check for silence to trigger processing
                    current_time = time.time()
                    if (current_time - self.last_processing_time > self.processing_cooldown and 
                        len(self.audio_buffer) >= int(self.SAMPLE_RATE * self.MIN_AUDIO_LENGTH)):
                        
                        # Calculate audio level in silence buffer
                        if len(self.silence_buffer) >= int(self.SAMPLE_RATE * self.SILENCE_DURATION):
                            recent_audio = list(self.silence_buffer)[-int(self.SAMPLE_RATE * self.SILENCE_DURATION):]
                            audio_level = np.sqrt(np.mean(np.array(recent_audio, dtype=np.float32) ** 2))
                            
                            # If silence detected and we have enough audio, process it
                            if audio_level < self.SILENCE_THRESHOLD * 32768:
                                # Extract all available audio for processing
                                stt_chunk = list(self.audio_buffer)
                                self.audio_buffer.clear()
                                self.silence_buffer.clear()
                                
                                # Convert to bytes and send to STT queue
                                stt_audio = np.array(stt_chunk, dtype=np.int16).tobytes()
                                self.audio_queue.put(stt_audio)
                                self.last_processing_time = current_time
                                
                                print(f"🔇 Silence detected, processing {len(stt_chunk)/self.SAMPLE_RATE:.1f}s of audio")
                        
                        # Fallback: if audio buffer gets too long without silence, process it anyway
                        elif len(self.audio_buffer) >= int(self.SAMPLE_RATE * 8.0):  # 8 seconds max
                            stt_chunk = list(self.audio_buffer)
                            self.audio_buffer.clear()
                            self.silence_buffer.clear()
                            
                            stt_audio = np.array(stt_chunk, dtype=np.int16).tobytes()
                            self.audio_queue.put(stt_audio)
                            self.last_processing_time = current_time
                            
                            print(f"⏰ Max duration reached, processing {len(stt_chunk)/self.SAMPLE_RATE:.1f}s of audio")
                        
                except Exception as e:
                    print(f"❌ Recording error: {e}")
                    break
            
            stream.stop_stream()
            stream.close()
            
        except Exception as e:
            print(f"❌ Failed to start recording: {e}")
    
    def _stt_processing_thread(self):
        """Thread for processing STT in chunks"""
        print("🔍 STT processing started...")
        
        while self.is_running:
            try:
                # Get audio chunk from queue
                audio_data = self.audio_queue.get(timeout=1.0)
                
                # Track STT latency
                stt_start_time = time.time()
                
                # Transcribe audio
                text = self._transcribe_audio(audio_data)
                
                stt_end_time = time.time()
                stt_latency = stt_end_time - stt_start_time
                self.latency_stats['stt_processing'].append(stt_latency)
                
                if text and text.strip():
                    print(f"📝 STT: '{text}' (latency: {stt_latency:.3f}s)")
                    self.tts_queue.put((text, stt_start_time))  # Pass start time for total pipeline tracking
                
            except queue.Empty:
                continue
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
        """Thread for processing TTS requests"""
        print("🎵 TTS processing started...")
        
        while self.is_running:
            try:
                # Get text from queue
                text_data = self.tts_queue.get(timeout=1.0)
                text, pipeline_start_time = text_data
                
                # Track TTS latency
                tts_start_time = time.time()
                
                # Convert to speech
                asyncio.run(self._text_to_speech_stream(text, pipeline_start_time, tts_start_time))
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"❌ TTS processing error: {e}")
    
    async def _text_to_speech_stream(self, text, pipeline_start_time, tts_start_time):
        """Convert text to speech using ElevenLabs WebSocket streaming"""
        if not self.ELEVENLABS_API_KEY:
            return
        
        try:
            uri = (
                f"wss://api.elevenlabs.io/v1/text-to-speech/{self.VOICE_ID}/stream-input"
                f"?model_id={self.MODEL_ID}&optimize_streaming_latency=4"
            )
            headers = {
                "xi-api-key": self.ELEVENLABS_API_KEY,
                "Content-Type": "application/json"
            }
            
            # Track network connection time
            network_start = time.time()
            async with websockets.connect(uri, extra_headers=headers) as websocket:
                network_connect_time = time.time() - network_start
                self.latency_stats['tts_network'].append(network_connect_time)
                
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
                
                # Receive and play audio chunks in real-time
                self.is_speaking = True
                first_audio_received = False
                async for message in websocket:
                    try:
                        data = json.loads(message)
                        
                        if "audio" in data and data["audio"]:
                            if not first_audio_received:
                                # Track time to first audio
                                tts_first_audio_time = time.time()
                                tts_latency = tts_first_audio_time - tts_start_time
                                self.latency_stats['tts_processing'].append(tts_latency)
                                first_audio_received = True
                            
                            audio_data = base64.b64decode(data["audio"])
                            self.playback_queue.put((audio_data, pipeline_start_time))
                        
                        if data.get("isFinal"):
                            break
                            
                    except Exception as e:
                        print(f"❌ TTS message error: {e}")
                        break
                
                self.is_speaking = False
                
        except Exception as e:
            print(f"❌ TTS streaming error: {e}")
            self.is_speaking = False
    
    def _audio_playback_thread(self):
        """Thread for playing TTS audio in real-time"""
        print("🔊 Audio playback started...")
        
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
                    # Get audio chunk from queue
                    audio_data = self.playback_queue.get(timeout=0.1)
                    mp3_data, pipeline_start_time = audio_data
                    
                    # Track total pipeline latency when first audio plays
                    playback_start_time = time.time()
                    total_latency = playback_start_time - pipeline_start_time
                    self.latency_stats['total_pipeline'].append(total_latency)
                    self.total_phrases_processed += 1
                    
                    print(f"🔊 Playing audio (total latency: {total_latency:.3f}s)")
                    
                    # Decode MP3 to PCM
                    from pydub import AudioSegment
                    import io
                    
                    audio_segment = AudioSegment.from_file(io.BytesIO(mp3_data), format="mp3")
                    pcm_data = audio_segment.get_array_of_samples()
                    
                    # Convert to float32 for PyAudio
                    pcm_float = np.array(pcm_data, dtype=np.float32) / 32768.0
                    
                    # Play in chunks
                    chunk_size = 1024
                    for i in range(0, len(pcm_float), chunk_size):
                        chunk = pcm_float[i:i + chunk_size]
                        if len(chunk) < chunk_size:
                            # Pad with zeros if needed
                            chunk = np.pad(chunk, (0, chunk_size - len(chunk)))
                        stream.write(chunk.astype(np.float32).tobytes())
                    
                    # Track playback duration
                    playback_end_time = time.time()
                    playback_duration = playback_end_time - playback_start_time
                    self.latency_stats['audio_playback'].append(playback_duration)
                        
                except queue.Empty:
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
    voice_changer = RealtimeVoiceChanger()
    voice_changer.start_streaming()

if __name__ == "__main__":
    main() 