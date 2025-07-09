#!/usr/bin/env python3
"""
Free Tier Optimized Voice Changer
Designed for ElevenLabs free tier limitations (2 concurrent requests)
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

class FreeTierVoiceChanger:
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
        
        # Free tier optimized settings
        self.BUFFER_DURATION = 3.0  # Longer buffers to reduce request frequency
        self.BUFFER_SIZE = int(self.SAMPLE_RATE * self.BUFFER_DURATION)
        self.OVERLAP_DURATION = 0.5
        self.OVERLAP_SIZE = int(self.SAMPLE_RATE * self.OVERLAP_DURATION)
        self.PROCESSING_INTERVAL = 0.5  # Slower processing to respect limits
        
        # Free tier limits
        self.MAX_CONCURRENT_REQUESTS = 2
        self.REQUEST_DELAY = 1.0  # 1 second between requests
        self.MAX_QUEUE_SIZE = 3  # Small queues to prevent buildup
        
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
            'tts_processing': [],
            'total_pipeline': []
        }
        self.session_start_time = None
        self.total_phrases_processed = 0
        
        # Audio buffer
        self.audio_buffer = deque(maxlen=int(self.SAMPLE_RATE * 8))
        self.last_processed_position = 0
        self.last_processing_time = 0
        
        # PyAudio instance
        self.audio = pyaudio.PyAudio()
        
        # Free tier connection management
        self.active_requests = 0
        self.last_request_time = 0
        self.request_lock = threading.Lock()
        
        # Initialize Google client
        self.speech_client = None
        if self.GOOGLE_APPLICATION_CREDENTIALS:
            self.speech_client = speech.SpeechClient()
    
    def start_streaming(self):
        """Start the free tier optimized voice changer"""
        if not self.ELEVENLABS_API_KEY:
            print("❌ ELEVENLABS_API_KEY not found")
            return False
        
        if not self.GOOGLE_APPLICATION_CREDENTIALS:
            print("❌ GOOGLE_APPLICATION_CREDENTIALS not found")
            return False
        
        print("🆓 Starting Free Tier Voice Changer...")
        print("   Optimized for ElevenLabs free tier (2 concurrent requests)")
        print("   Respects rate limits and concurrency restrictions")
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
            print("\n🛑 Stopping free tier voice changer...")
            self.is_running = False
            self._print_latency_report()
        
        recording_thread.join()
        stt_thread.join()
        tts_thread.join()
        playback_thread.join()
        
        self.audio.terminate()
        print("✅ Free tier voice changer stopped")
    
    def _print_latency_report(self):
        """Print comprehensive latency report"""
        if not self.latency_stats['total_pipeline']:
            print("📊 No latency data collected")
            return
        
        print("\n" + "="*60)
        print("📊 FREE TIER VOICE CHANGER PERFORMANCE REPORT")
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
            
            print(f"\n🆓 FREE TIER OPTIMIZATIONS:")
            print(f"Max Concurrent Requests: {self.MAX_CONCURRENT_REQUESTS}")
            print(f"Request Delay: {self.REQUEST_DELAY}s")
            print(f"Buffer Duration: {self.BUFFER_DURATION}s")
            print(f"Processing Interval: {self.PROCESSING_INTERVAL}s")
        
        print("="*60)
    
    def _audio_recording_thread(self):
        """Free tier optimized audio recording"""
        try:
            stream = self.audio.open(
                format=self.FORMAT,
                channels=self.CHANNELS,
                rate=self.SAMPLE_RATE,
                input=True,
                frames_per_buffer=self.CHUNK_SIZE
            )
            
            print("🎤 Free tier recording started...")
            
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
        """Free tier optimized STT processing"""
        print("🔍 Free tier STT processing started...")
        
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
        """Free tier optimized TTS processing with rate limiting"""
        print("🎵 Free tier TTS processing started...")
        
        while self.is_running:
            try:
                # Non-blocking queue get with timeout
                try:
                    text_data = self.tts_queue.get(timeout=1.0)
                except queue.Empty:
                    continue
                
                text, pipeline_start_time = text_data
                tts_start_time = time.time()
                
                # Wait for available request slot
                with self.request_lock:
                    while self.active_requests >= self.MAX_CONCURRENT_REQUESTS:
                        print("⏳ Waiting for available request slot...")
                        time.sleep(0.5)
                    
                    # Respect rate limiting
                    current_time = time.time()
                    time_since_last = current_time - self.last_request_time
                    if time_since_last < self.REQUEST_DELAY:
                        sleep_time = self.REQUEST_DELAY - time_since_last
                        print(f"⏳ Rate limiting: waiting {sleep_time:.1f}s")
                        time.sleep(sleep_time)
                    
                    self.active_requests += 1
                    self.last_request_time = time.time()
                
                # Convert to speech using REST API (more reliable for free tier)
                success = self._tts_rest_api(text, pipeline_start_time, tts_start_time)
                
                # Release request slot
                with self.request_lock:
                    self.active_requests -= 1
                
                if not success:
                    print("⚠️ TTS failed, skipping text")
                
            except Exception as e:
                print(f"❌ TTS processing error: {e}")
                with self.request_lock:
                    self.active_requests = max(0, self.active_requests - 1)
    
    def _tts_rest_api(self, text, pipeline_start_time, tts_start_time):
        """TTS using REST API (optimized for free tier)"""
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
            
            # Make REST API request with timeout
            response = requests.post(url, json=data, headers=headers, timeout=30)
            
            if response.status_code == 200:
                # Track REST API latency
                tts_end_time = time.time()
                tts_latency = tts_end_time - tts_start_time
                self.latency_stats['tts_processing'].append(tts_latency)
                
                # Add to playback queue
                try:
                    self.playback_queue.put_nowait((response.content, pipeline_start_time))
                    print(f"✅ REST API TTS successful (latency: {tts_latency:.3f}s)")
                    return True
                except queue.Full:
                    print("⚠️ Playback queue full, skipping audio")
                    return False
            elif response.status_code == 429:
                print("⚠️ Rate limited (429), will retry later")
                return False
            else:
                print(f"❌ REST API error: {response.status_code} - {response.text}")
                return False
                
        except requests.exceptions.Timeout:
            print("⚠️ REST API timeout")
            return False
        except Exception as e:
            print(f"❌ REST API TTS error: {e}")
            return False
    
    def _audio_playback_thread(self):
        """Free tier optimized audio playback"""
        print("🔊 Free tier audio playback started...")
        
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
    voice_changer = FreeTierVoiceChanger()
    voice_changer.start_streaming()

if __name__ == "__main__":
    main() 