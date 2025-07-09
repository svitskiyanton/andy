#!/usr/bin/env python3
"""
Multi-Process Voice Changer
STT and TTS run in separate processes with inter-process communication
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
import multiprocessing as mp
from multiprocessing import Queue, Process, Lock
import pickle

# Load environment variables
load_dotenv()

class MultiProcessVoiceChanger:
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
        
        # Multi-process settings
        self.COLLECTION_WINDOW = 1.5  # Initial buffer time
        self.MIN_PHRASE_LENGTH = 3    # Minimum characters to process
        self.MAX_PHRASE_LENGTH = 200  # Maximum characters per phrase
        
        # Control flags
        self.is_running = False
        
        # Inter-process queues
        self.audio_queue = Queue()      # Audio chunks → STT process
        self.stt_queue = Queue()        # STT results → Main process
        self.tts_queue = Queue()        # Phrases → TTS process
        self.audio_output_queue = Queue()  # Audio data → Audio process
        
        # Process locks
        self.stt_lock = Lock()
        self.tts_lock = Lock()
        
        # Latency tracking
        self.latency_stats = {
            'stt_processing': [],
            'tts_processing': [],
            'total_pipeline': [],
            'inter_process_latency': []
        }
        self.session_start_time = None
        self.total_phrases_processed = 0
        
        # PyAudio instance
        self.audio = pyaudio.PyAudio()
        
        # Process references
        self.stt_process = None
        self.tts_process = None
        self.audio_process = None
    
    def start_streaming(self):
        """Start the multi-process voice changer"""
        if not self.ELEVENLABS_API_KEY:
            print("❌ ELEVENLABS_API_KEY not found")
            return False
        
        if not self.GOOGLE_APPLICATION_CREDENTIALS:
            print("❌ GOOGLE_APPLICATION_CREDENTIALS not found")
            return False
        
        print("🚀 Starting Multi-Process Voice Changer...")
        print("   STT runs in separate process")
        print("   TTS runs in separate process")
        print("   Audio runs in separate process")
        print("   Inter-process communication via queues")
        print("   Press Ctrl+C to stop")
        
        self.session_start_time = time.time()
        self.is_running = True
        
        try:
            # Start all processes
            self._start_processes()
            
            # Run main process loop
            self._main_process_loop()
            
        except KeyboardInterrupt:
            print("\n🛑 Stopping multi-process voice changer...")
            self.is_running = False
            self._stop_processes()
            self._print_latency_report()
        
        self.audio.terminate()
        print("✅ Multi-process voice changer stopped")
    
    def _start_processes(self):
        """Start STT, TTS, and Audio processes"""
        print("🚀 Starting processes...")
        
        # Start STT process
        self.stt_process = Process(
            target=stt_process_worker,
            args=(self.audio_queue, self.stt_queue, self.stt_lock)
        )
        self.stt_process.start()
        print("✅ STT process started")
        
        # Start TTS process
        self.tts_process = Process(
            target=tts_process_worker,
            args=(self.tts_queue, self.audio_output_queue, self.tts_lock)
        )
        self.tts_process.start()
        print("✅ TTS process started")
        
        # Start Audio process
        self.audio_process = Process(
            target=audio_process_worker,
            args=(self.audio_output_queue,)
        )
        self.audio_process.start()
        print("✅ Audio process started")
    
    def _stop_processes(self):
        """Stop all processes gracefully"""
        print("🛑 Stopping processes...")
        
        # Send stop signals
        self.audio_queue.put(None)  # Stop STT
        self.tts_queue.put(None)    # Stop TTS
        self.audio_output_queue.put(None)  # Stop Audio
        
        # Wait for processes to finish
        if self.stt_process:
            self.stt_process.join(timeout=2)
            if self.stt_process.is_alive():
                self.stt_process.terminate()
        
        if self.tts_process:
            self.tts_process.join(timeout=2)
            if self.tts_process.is_alive():
                self.tts_process.terminate()
        
        if self.audio_process:
            self.audio_process.join(timeout=2)
            if self.audio_process.is_alive():
                self.audio_process.terminate()
        
        print("✅ All processes stopped")
    
    def _main_process_loop(self):
        """Main process loop - coordinates between processes"""
        print("🎯 Main process loop started...")
        print(f"⏰ Initial buffer time: {self.COLLECTION_WINDOW}s")
        
        # Wait for initial buffer
        time.sleep(self.COLLECTION_WINDOW)
        print(f"✅ Initial buffer filled ({self.COLLECTION_WINDOW}s)")
        
        # Start audio recording
        self._start_audio_recording()
        
        # Main coordination loop
        current_phrase = ""
        last_stt_time = 0
        
        while self.is_running:
            try:
                # Check for STT results
                if not self.stt_queue.empty():
                    stt_result = self.stt_queue.get_nowait()
                    
                    if stt_result and stt_result.get('text'):
                        text = stt_result['text']
                        timestamp = stt_result['timestamp']
                        
                        # Add to current phrase
                        if current_phrase:
                            current_phrase += " " + text
                        else:
                            current_phrase = text
                        last_stt_time = timestamp
                        
                        print(f"📝 STT: '{text}' (latency: {stt_result.get('latency', 0):.3f}s)")
                
                # Process current phrase
                if current_phrase and time.time() - last_stt_time > 0.5:
                    # Check phrase length
                    if len(current_phrase) >= self.MIN_PHRASE_LENGTH:
                        print(f"📦 Processing phrase: '{current_phrase}'")
                        
                        # Send to TTS process
                        self.tts_queue.put({
                            'text': current_phrase,
                            'timestamp': time.time()
                        })
                        
                        self.total_phrases_processed += 1
                    
                    current_phrase = ""
                
                # Small delay
                time.sleep(0.01)
                
            except Exception as e:
                print(f"❌ Main process error: {e}")
                break
    
    def _start_audio_recording(self):
        """Start audio recording in main process"""
        print("🎤 Audio recording started...")
        
        try:
            stream = self.audio.open(
                format=self.FORMAT,
                channels=self.CHANNELS,
                rate=self.SAMPLE_RATE,
                input=True,
                frames_per_buffer=self.CHUNK_SIZE
            )
            
            # Process in 2-second chunks
            chunk_duration = 2.0
            chunk_samples = int(self.SAMPLE_RATE * chunk_duration)
            current_chunk = []
            
            while self.is_running:
                try:
                    # Read audio chunk
                    data = stream.read(self.CHUNK_SIZE, exception_on_overflow=False)
                    audio_chunk = np.frombuffer(data, dtype=np.int16)
                    
                    # Add to current chunk
                    current_chunk.extend(audio_chunk)
                    
                    # When chunk is full, send to STT process
                    if len(current_chunk) >= chunk_samples:
                        # Send to STT process
                        self.audio_queue.put(list(current_chunk))
                        current_chunk = []
                    
                    # Small delay
                    time.sleep(0.01)
                    
                except Exception as e:
                    print(f"❌ Recording error: {e}")
                    break
            
            stream.stop_stream()
            stream.close()
            
        except Exception as e:
            print(f"❌ Failed to start recording: {e}")
    
    def _print_latency_report(self):
        """Print comprehensive latency report"""
        if not self.latency_stats['total_pipeline']:
            print("📊 No latency data collected")
            return
        
        print("\n" + "="*60)
        print("📊 MULTI-PROCESS VOICE CHANGER PERFORMANCE REPORT")
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

def stt_process_worker(audio_queue, stt_queue, lock):
    """STT process worker - runs in separate process"""
    print("🔍 STT process worker started...")
    
    # Initialize Google client in this process
    speech_client = None
    if os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
        try:
            speech_client = speech.SpeechClient()
            print("✅ Google STT client initialized")
        except Exception as e:
            print(f"❌ Failed to initialize Google STT: {e}")
            return
    
    while True:
        try:
            # Get audio chunk from queue
            audio_chunk = audio_queue.get(timeout=1.0)
            
            # Check for stop signal
            if audio_chunk is None:
                print("🛑 STT process received stop signal")
                break
            
            # Process STT
            stt_start = time.time()
            text = transcribe_audio_sync(audio_chunk, speech_client)
            stt_end = time.time()
            
            if text and text.strip():
                stt_latency = stt_end - stt_start
                
                # Send result to main process
                stt_queue.put({
                    'text': text.strip(),
                    'latency': stt_latency,
                    'timestamp': stt_end
                })
                
                print(f"🔍 STT: '{text.strip()}' ({stt_latency:.3f}s)")
            
        except Exception as e:
            print(f"❌ STT process error: {e}")
            continue
    
    print("✅ STT process worker stopped")

def tts_process_worker(tts_queue, audio_output_queue, lock):
    """TTS process worker - runs in separate process, streams audio from ElevenLabs"""
    print("🎵 TTS process worker started...")
    import websockets
    import asyncio
    import base64
    import json
    import time
    import os
    from pydub import AudioSegment
    import io

    api_key = os.getenv("ELEVENLABS_API_KEY")
    voice_id = os.getenv("VOICE_ID", "21m00Tcm4TlvDq8ikWAM")
    model_id = "eleven_multilingual_v2"

    async def tts_stream(text, audio_output_queue):
        uri = (
            f"wss://api.elevenlabs.io/v1/text-to-speech/{voice_id}/stream-input"
            f"?model_id={model_id}&optimize_streaming_latency=4"
        )
        headers = {
            "xi-api-key": api_key,
            "Content-Type": "application/json"
        }
        try:
            async with websockets.connect(uri, extra_headers=headers) as websocket:
                print("✅ TTS WebSocket connected")
                
                # Send initialization
                init_message = {
                    "text": " ",
                    "voice_settings": {
                        "speed": 1,
                        "stability": 0.5,
                        "similarity_boost": 0.8
                    },
                    "xi_api_key": api_key
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
                
                # Receive and forward audio chunks
                audio_chunks = 0
                async for message in websocket:
                    data = json.loads(message)
                    if "audio" in data and data["audio"]:
                        audio_data = base64.b64decode(data["audio"])
                        audio_chunks += 1
                        print(f"🎵 TTS audio chunk {audio_chunks}: {len(audio_data)} bytes")
                        # Forward to audio process
                        audio_output_queue.put(audio_data)
                    if data.get("isFinal"):
                        print(f"✅ TTS stream completed ({audio_chunks} chunks)")
                        break
        except Exception as e:
            print(f"❌ TTS streaming error: {e}")

    while True:
        try:
            phrase_data = tts_queue.get(timeout=1.0)
            if phrase_data is None:
                print("🛑 TTS process received stop signal")
                break
            text = phrase_data['text']
            print(f"🎵 TTS processing: '{text}'")
            asyncio.run(tts_stream(text, audio_output_queue))
        except Exception as e:
            print(f"❌ TTS process error: {e}")
            continue
    print("✅ TTS process worker stopped")

def audio_process_worker(audio_output_queue):
    """Audio process worker - runs in separate process, decodes and plays MP3 audio"""
    print("🔊 Audio process worker started...")
    import pyaudio
    import numpy as np
    from pydub import AudioSegment
    import io
    audio = pyaudio.PyAudio()
    stream = None
    try:
        while True:
            try:
                audio_data = audio_output_queue.get(timeout=1.0)
                if audio_data is None:
                    print("🛑 Audio process received stop signal")
                    break
                
                # Validate audio data
                if len(audio_data) < 100:
                    continue
                
                # Decode MP3 to PCM
                audio_segment = AudioSegment.from_file(io.BytesIO(audio_data), format="mp3")
                if len(audio_segment) == 0:
                    continue
                
                pcm_data = audio_segment.get_array_of_samples()
                pcm_float = np.array(pcm_data, dtype=np.float32) / 32768.0
                
                # Open stream if not already
                if stream is None:
                    stream = audio.open(format=pyaudio.paFloat32, channels=1, rate=44100, output=True, frames_per_buffer=1024)
                    print("🔊 Audio playback stream initialized")
                
                # Play in chunks
                chunk_size = 1024
                for i in range(0, len(pcm_float), chunk_size):
                    chunk = pcm_float[i:i + chunk_size]
                    if len(chunk) < chunk_size:
                        chunk = np.pad(chunk, (0, chunk_size - len(chunk)))
                    stream.write(chunk.astype(np.float32).tobytes())
                
                print(f"🔊 Audio played: {len(audio_data)} bytes → {len(pcm_float)} samples")
                
            except Exception as e:
                print(f"❌ Audio process error: {e}")
                continue
    finally:
        if stream:
            stream.stop_stream()
            stream.close()
        audio.terminate()
        print("✅ Audio process worker stopped")

def transcribe_audio_sync(audio_chunk, speech_client):
    """Synchronous transcription (runs in STT process)"""
    try:
        audio_data = np.array(audio_chunk, dtype=np.int16).tobytes()
        
        audio = speech.RecognitionAudio(content=audio_data)
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=16000,
            language_code="ru-RU",
            enable_automatic_punctuation=True,
            model="latest_long"
        )
        
        response = speech_client.recognize(config=config, audio=audio)
        
        transcribed_text = ""
        for result in response.results:
            transcribed_text += result.alternatives[0].transcript + " "
        
        return transcribed_text.strip()
        
    except Exception as e:
        print(f"❌ Sync transcription error: {e}")
        return None

def main():
    """Main function"""
    # Set multiprocessing start method
    mp.set_start_method('spawn', force=True)
    
    voice_changer = MultiProcessVoiceChanger()
    voice_changer.start_streaming()

if __name__ == "__main__":
    main() 