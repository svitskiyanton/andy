#!/usr/bin/env python3
"""
Improved live playback system that eliminates gaps
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

class SmoothLivePlayback:
    def __init__(self):
        # Audio settings
        self.FORMAT = pyaudio.paFloat32
        self.CHANNELS = 1
        self.SAMPLE_RATE = 44100
        self.CHUNK_SIZE = 1024
        
        # Buffer settings
        self.BUFFER_SIZE = 15  # Increased buffer size
        self.MIN_BUFFER_SIZE = 5
        self.MAX_BUFFER_SIZE = 30
        
        # Audio objects
        self.audio = pyaudio.PyAudio()
        self.audio_buffer = deque(maxlen=self.MAX_BUFFER_SIZE)
        self.buffer_lock = threading.Lock()
        
        # Playback control
        self.playback_started = False
        self.streaming_complete = False
        self.mp3_chunks = []
        
        # Output file
        self.OUTPUT_FILE = "smooth_live_output.mp3"
        
        # Continuous playback buffer
        self.continuous_buffer = queue.Queue(maxsize=100)
        self.playback_thread = None
        
    async def stream_text_to_speech_smooth(self):
        """Stream text to speech with smooth live playback"""
        # Test text (same as before)
        self.test_text = """- Между пищеблоком и узкоколейкой.
     Я хотел спросить: "А где  узкоколейка?" - но передумал. Торопиться  мне
было некуда. Найду.
     Выяснилось, что комбинат занимает огромную территорию. К югу он тянулся
до станции Пискаревка. Северная его граница проходила вдоль безымянной реки.
     Короче,  я  довольно  быстро  заблудился.  Среди  одинаковых  кирпичных
пакгаузов бродили люди. Я спрашивал у некоторых - где четвертый холодильник?
Ответы звучали невнятно и рассеянно. Позднее я узнал, что на этой базе царит
тотальное государственное хищение в особо крупных  размерах.  Крали все. Все
без исключения. И потому у всех были такие отрешенные, задумчивые лица.
     Фрукты уносили в карманах и за пазухой. В подвязанных снизу  шароварах.
В  футлярах  от  музыкальных   инструментов.   Набивали   ими  вместительные
учрежденческие портфели."""
        
        print("🎵 Starting Smooth Live Playback Test...")
        print("=" * 80)
        print("Text to stream:")
        print(self.test_text)
        print("=" * 80)
        
        # Split text into chunks
        text_chunks = self._split_text_into_chunks(self.test_text, max_chunk_size=100)
        print(f"📦 Buffer settings: {self.BUFFER_SIZE} chunks to start, {self.MIN_BUFFER_SIZE}-{self.MAX_BUFFER_SIZE} range")
        
        # Connect to ElevenLabs
        import websockets
        uri = "wss://api.elevenlabs.io/v1/text-to-speech/GN4wbsbejSnGSa1AzjH5/stream-input?model_id=eleven_multilingual_v2&optimize_streaming_latency=4"
        
        print(f"🔗 Connecting to: {uri}")
        print("⏳ Establishing WebSocket connection...")
        
        start_time = time.time()
        async with websockets.connect(uri) as websocket:
            connection_time = time.time() - start_time
            print(f"✅ WebSocket connected (latency: {connection_time:.3f}s)")
            
            # Get API key
            api_key = self._get_api_key()
            if not api_key:
                print("❌ API key not found. Exiting.")
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
            print("📤 Sent initialization message")
            
            # Send text chunks
            print(f"📝 Sending {len(text_chunks)} text chunks...")
            for i, chunk in enumerate(text_chunks, 1):
                message = {
                    "text": chunk,
                    "xi_api_key": api_key
                }
                await websocket.send(json.dumps(message))
                print(f"📤 Sent chunk {i}/{len(text_chunks)}: '{chunk[:30]}{'...' if len(chunk) > 30 else ''}' ({len(chunk)} chars)")
            
            # Send end marker
            end_message = {
                "text": "",
                "xi_api_key": api_key
            }
            await websocket.send(json.dumps(end_message))
            print("📤 Sent end marker")
            
            # Start smooth audio streaming
            print("🔊 Starting smooth audio streaming...")
            await self._smooth_audio_streaming(websocket)
    
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
            
            print("🎵 Audio stream initialized")
            print(f"💾 Will save to: {self.OUTPUT_FILE}")
            
            # Start continuous playback thread
            self.playback_thread = threading.Thread(target=self._continuous_playback_worker, args=(stream,))
            self.playback_thread.daemon = True
            self.playback_thread.start()
            
            # Track metrics
            first_audio_time = None
            total_audio_chunks = 0
            total_audio_bytes = 0
            streaming_start = time.time()
            
            # Receive and process audio chunks
            async for message in websocket:
                try:
                    data = json.loads(message)
                    
                    if "audio" in data and data["audio"]:
                        # Track first audio timing
                        if first_audio_time is None:
                            first_audio_time = time.time()
                            time_to_first_audio = first_audio_time - streaming_start
                            print(f"🎵 First audio received (latency: {time_to_first_audio:.3f}s)")
                        
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
                        
                        # Start playback when we have enough data
                        if not self.playback_started and total_audio_chunks >= self.BUFFER_SIZE:
                            self.playback_started = True
                            print(f"🎵 Starting continuous playback (buffer: {total_audio_chunks} chunks)")
                        
                        # Show progress
                        if total_audio_chunks <= 5 or total_audio_chunks % 10 == 0:
                            print(f"🎵 Audio chunk {total_audio_chunks}: {len(audio_data):,} bytes")
                    
                    elif "audio" in data and data["audio"] is None:
                        print("📡 End of stream signal received")
                    
                    if data.get("isFinal"):
                        print("✅ Stream completed")
                        self.streaming_complete = True
                        break
                        
                except json.JSONDecodeError as e:
                    print(f"⚠️ JSON decode error: {e}")
                    continue
                except Exception as e:
                    print(f"❌ Message processing error: {e}")
                    break
            
            # Wait for playback to complete
            while not self.continuous_buffer.empty():
                await asyncio.sleep(0.1)
            
            # Save MP3 file
            if self.mp3_chunks:
                print(f"💾 Saving {len(self.mp3_chunks)} audio chunks to {self.OUTPUT_FILE}...")
                await self._save_mp3_file(self.mp3_chunks)
            
            # Final metrics
            streaming_end = time.time()
            total_streaming_time = streaming_end - streaming_start
            
            print("\n" + "=" * 60)
            print("🎵 SMOOTH LIVE PLAYBACK METRICS:")
            print("=" * 60)
            print(f"Total streaming time: {total_streaming_time:.3f}s")
            print(f"Time to first audio: {time_to_first_audio:.3f}s")
            print(f"Total audio chunks: {total_audio_chunks}")
            print(f"Total audio bytes: {total_audio_bytes:,}")
            print(f"Average chunk size: {total_audio_bytes/total_audio_chunks:.0f} bytes")
            print(f"Streaming rate: {total_audio_bytes/total_streaming_time:.0f} bytes/s")
            print(f"Output file: {self.OUTPUT_FILE}")
            print("=" * 60)
            
            # Cleanup
            stream.stop_stream()
            stream.close()
            
        except Exception as e:
            print(f"❌ Smooth streaming error: {e}")
            import traceback
            traceback.print_exc()
    
    def _continuous_playback_worker(self, stream):
        """Continuous playback worker that eliminates gaps"""
        try:
            while not self.streaming_complete or not self.continuous_buffer.empty():
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
            # Quick validation
            if len(mp3_data) < 100:
                return
            
            # Decode MP3 to PCM (optimized for speed)
            try:
                audio_segment = AudioSegment.from_file(io.BytesIO(mp3_data), format="mp3")
            except Exception:
                # Skip problematic chunks to maintain continuity
                return
            
            if len(audio_segment) == 0:
                return
            
            # Convert to PCM (optimized)
            pcm_data = audio_segment.get_array_of_samples()
            pcm_float = np.array(pcm_data, dtype=np.float32) / 32768.0
            
            # Play immediately without additional delays
            stream.write(pcm_float.astype(np.float32).tobytes())
            
        except Exception as e:
            # Silent error handling to avoid gaps
            pass
    
    async def _save_mp3_file(self, mp3_chunks):
        """Save MP3 chunks to file"""
        try:
            from pydub import AudioSegment
            import io
            
            print(f"🔧 Processing {len(mp3_chunks)} MP3 chunks...")
            
            # Process chunks
            audio_segments = []
            for i, mp3_chunk in enumerate(mp3_chunks):
                try:
                    chunk_audio = AudioSegment.from_file(io.BytesIO(mp3_chunk), format="mp3")
                    if chunk_audio.frame_rate != 44100:
                        chunk_audio = chunk_audio.set_frame_rate(44100)
                    if chunk_audio.channels != 1:
                        chunk_audio = chunk_audio.set_channels(1)
                    audio_segments.append(chunk_audio)
                except Exception:
                    continue
            
            if audio_segments:
                print(f"🔧 Combining {len(audio_segments)} audio segments...")
                
                # Combine with crossfading
                combined_audio = audio_segments[0]
                for i in range(1, len(audio_segments)):
                    combined_audio = combined_audio.append(audio_segments[i], crossfade=50)
                
                # Export
                print(f"🔧 Exporting to {self.OUTPUT_FILE}...")
                combined_audio.export(self.OUTPUT_FILE, format="mp3", bitrate="192k")
                
                # File info
                file_size = os.path.getsize(self.OUTPUT_FILE)
                duration = len(combined_audio) / 1000.0
                
                print(f"✅ Successfully saved to {self.OUTPUT_FILE}")
                print(f"📊 File size: {file_size:,} bytes")
                print(f"⏱️  Duration: {duration:.2f} seconds")
                
        except Exception as e:
            print(f"❌ Error saving MP3 file: {e}")

    def _get_api_key(self):
        """Get API key from environment, file, or user input"""
        # Try environment variable first
        api_key = os.getenv("ELEVENLABS_API_KEY")
        if api_key:
            return api_key
        
        # Try reading from .env file
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
        
        # Try reading from env_template.txt
        try:
            if os.path.exists("env_template.txt"):
                with open("env_template.txt", "r") as f:
                    for line in f:
                        if line.startswith("ELEVENLABS_API_KEY="):
                            api_key = line.split("=", 1)[1].strip()
                            if api_key and api_key != "your_api_key_here":
                                return api_key
        except Exception:
            pass
        
        # Prompt user to enter API key
        print("🔑 ElevenLabs API Key not found.")
        print("Please enter your ElevenLabs API key:")
        api_key = input("API Key: ").strip()
        
        if api_key:
            # Save to .env file for future use
            try:
                with open(".env", "w") as f:
                    f.write(f"ELEVENLABS_API_KEY={api_key}\n")
                print("✅ API key saved to .env file for future use")
            except Exception:
                print("⚠️ Could not save API key to file")
            
            return api_key
        
        return None

async def main():
    """Main function"""
    playback = SmoothLivePlayback()
    await playback.stream_text_to_speech_smooth()

if __name__ == "__main__":
    asyncio.run(main()) 