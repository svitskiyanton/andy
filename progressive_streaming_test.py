#!/usr/bin/env python3
"""
Progressive WebSocket Streaming Test for ElevenLabs TTS
Implements smooth streaming playback like browser MP3 players
"""

import os
import asyncio
import json
import base64
import websockets
import pyaudio
import numpy as np
from dotenv import load_dotenv
import time
from collections import deque
import threading
import queue

# Load environment variables
load_dotenv()

class ProgressiveWebSocketStreamingTest:
    def __init__(self):
        self.ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
        self.VOICE_ID = os.getenv("VOICE_ID", "21m00Tcm4TlvDq8ikWAM")
        self.MODEL_ID = "eleven_multilingual_v2"
        
        # Audio settings
        self.SAMPLE_RATE = 44100
        self.CHANNELS = 1
        self.FORMAT = pyaudio.paFloat32
        self.CHUNK_SIZE = 1024
        
        # Progressive streaming settings
        self.BUFFER_SIZE = 10 # Number of audio chunks to buffer before starting playback
        self.MIN_BUFFER_SIZE = 3  # Minimum chunks to keep in buffer
        self.MAX_BUFFER_SIZE = 20  # Maximum chunks in buffer
        
        # PyAudio instance
        self.audio = pyaudio.PyAudio()
        
        # Output file settings
        self.OUTPUT_FILE = "progressive_streaming_output.mp3"
        
        # Progressive streaming buffers
        self.audio_buffer = deque(maxlen=self.MAX_BUFFER_SIZE)
        self.mp3_chunks = []  # For saving to file
        self.buffer_lock = threading.Lock()
        self.playback_started = False
        self.streaming_complete = False
        self.playback_queue = queue.Queue()
        
        # Test text (longer Russian text)
        self.test_text = """
Так почему же вы здесь?
     - А где же мне быть? Где же мне работать, по-твоему? В школе? Что я там
буду воровать, промокашки?!  Устраиваясь на работу, ты  должен  прежде всего
задуматься: что, где и как? Что я смогу украсть? Где  я смогу украсть? И как
я  смогу украсть?.. Ты  понял?  Вот и хорошо. Все будет нормально.  К вечеру
бабки появятся.
"""
    
    async def stream_text_to_speech_progressive(self):
        """Stream text to speech with progressive buffering"""
        if not self.ELEVENLABS_API_KEY:
            print("❌ ELEVENLABS_API_KEY not found")
            return
        
        print("🎵 Starting Progressive WebSocket Streaming Test...")
        print("=" * 60)
        print("Text to stream:")
        print(self.test_text[:100] + "..." if len(self.test_text) > 100 else self.test_text)
        print("=" * 60)
        print(f"�� Buffer settings: {self.BUFFER_SIZE} chunks to start, {self.MIN_BUFFER_SIZE}-{self.MAX_BUFFER_SIZE} range")
        
        try:
            # WebSocket connection
            uri = (
                f"wss://api.elevenlabs.io/v1/text-to-speech/{self.VOICE_ID}/stream-input"
                f"?model_id={self.MODEL_ID}&optimize_streaming_latency=4"
            )
            headers = {
                "xi-api-key": self.ELEVENLABS_API_KEY,
                "Content-Type": "application/json"
            }
            
            print(f"🔗 Connecting to: {uri}")
            print("⏳ Establishing WebSocket connection...")
            
            connection_start = time.time()
            async with websockets.connect(uri, extra_headers=headers) as websocket:
                connection_time = time.time() - connection_start
                print(f"✅ WebSocket connected (latency: {connection_time:.3f}s)")
                
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
                print("📤 Sent initialization message")
                
                # Send text in chunks for streaming
                text_chunks = self._split_text_into_chunks(self.test_text, max_chunk_size=100)
                print(f"📝 Sending {len(text_chunks)} text chunks...")
                
                # Debug: Track text chunking
                print("🔍 DEBUG: Progressive Streaming Text Analysis")
                print("=" * 60)
                print(f"📄 Total text length: {len(self.test_text):,} characters")
                print(f"📦 Chunks to send: {len(text_chunks)}")
                
                # Analyze chunks being sent
                chunk_sizes = [len(chunk) for chunk in text_chunks]
                print(f"📊 Chunk size stats:")
                print(f"   Smallest: {min(chunk_sizes)} characters")
                print(f"   Largest: {max(chunk_sizes)} characters")
                print(f"   Average: {sum(chunk_sizes) / len(chunk_sizes):.1f} characters")
                
                # Verify character count
                total_chars_sent = sum(len(chunk) for chunk in text_chunks)
                print(f"🔢 Character verification:")
                print(f"   Original text: {len(self.test_text):,} characters")
                print(f"   In chunks: {total_chars_sent:,} characters")
                print(f"   Difference: {len(self.test_text) - total_chars_sent} characters")
                
                if total_chars_sent != len(self.test_text):
                    print("⚠️ WARNING: Character count mismatch in progressive streaming!")
                else:
                    print("✅ Character count: Perfect match")
                print("=" * 60)
                
                # Track sent chunks
                sent_chunks = []
                
                for i, chunk in enumerate(text_chunks):
                    text_message = {
                        "text": chunk,
                        "try_trigger_generation": True
                    }
                    await websocket.send(json.dumps(text_message))
                    
                    # Debug: Track each chunk sent
                    sent_chunks.append(chunk)
                    print(f"📤 Sent chunk {i+1}/{len(text_chunks)}: '{chunk[:30]}...' ({len(chunk)} chars)")
                    
                    await asyncio.sleep(0.1)  # Small delay between chunks
                
                # Debug: Final sending summary
                print("🔍 DEBUG: Sending Summary")
                print("=" * 60)
                print(f"📤 Total chunks sent: {len(sent_chunks)}")
                print(f"📊 Total characters sent: {sum(len(chunk) for chunk in sent_chunks):,}")
                print(f"📄 Original text length: {len(self.test_text):,}")
                
                # Show first and last chunks sent
                if sent_chunks:
                    print(f"📋 First chunk sent: '{sent_chunks[0][:50]}{'...' if len(sent_chunks[0]) > 50 else ''}'")
                    print(f"📋 Last chunk sent: '{sent_chunks[-1][:50]}{'...' if len(sent_chunks[-1]) > 50 else ''}'")
                print("=" * 60)
                
                # Send end marker
                await websocket.send(json.dumps({"text": ""}))
                print("�� Sent end marker")
                
                # Start progressive audio streaming
                print("🔊 Starting progressive audio streaming...")
                await self._progressive_streaming_audio(websocket)
                
        except Exception as e:
            print(f"❌ WebSocket streaming error: {e}")
            import traceback
            traceback.print_exc()
    
    def _split_text_into_chunks(self, text, max_chunk_size=100):
        """Split text into chunks for streaming while preserving words and spacing"""
        chunks = []
        current_chunk = ""
        current_length = 0
        
        # Split by words but preserve the original spacing
        import re
        # Split by word boundaries but keep the separators
        parts = re.split(r'(\s+)', text)
        
        for part in parts:
            # If adding this part would exceed the limit
            if current_length + len(part) > max_chunk_size and current_chunk:
                # Save current chunk and start new one
                chunks.append(current_chunk)
                current_chunk = part
                current_length = len(part)
            else:
                # Add to current chunk
                current_chunk += part
                current_length += len(part)
        
        # Add the last chunk if it exists
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks
    
    async def _progressive_streaming_audio(self, websocket):
        """Progressive streaming with buffering - like browser MP3 players"""
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
            
            # Track streaming metrics
            first_audio_time = None
            total_audio_chunks = 0
            total_audio_bytes = 0
            streaming_start = time.time()
            
            # Debug: Track audio reception
            received_audio_chunks = []
            audio_chunk_sizes = []
            
            # Start playback task
            playback_task = asyncio.create_task(self._playback_worker(stream))
            
            # Receive and buffer audio chunks
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
                        
                        # Debug: Track audio chunks
                        received_audio_chunks.append(audio_data)
                        audio_chunk_sizes.append(len(audio_data))
                        
                        # Debug: Show audio chunk info
                        if total_audio_chunks <= 5 or total_audio_chunks % 10 == 0:
                            print(f"🎵 Audio chunk {total_audio_chunks}: {len(audio_data):,} bytes")
                        
                        # Add to buffer
                        with self.buffer_lock:
                            self.audio_buffer.append(audio_data)
                            buffer_size = len(self.audio_buffer)
                        
                        # Collect for file saving
                        self.mp3_chunks.append(audio_data)
                        
                        # Start playback when buffer is ready
                        if not self.playback_started and buffer_size >= self.BUFFER_SIZE:
                            self.playback_started = True
                            print(f"🎵 Starting playback (buffer: {buffer_size} chunks)")
                        
                        print(f"📦 Buffer: {buffer_size}/{self.MAX_BUFFER_SIZE} chunks")
                    
                    elif "audio" in data and data["audio"] is None:
                        print("📡 End of stream signal received")
                    
                    if data.get("isFinal"):
                        print("✅ Stream completed")
                        self.streaming_complete = True
                        
                        # Debug: Audio reception summary
                        print("🔍 DEBUG: Audio Reception Summary")
                        print("=" * 60)
                        print(f"🎵 Total audio chunks received: {total_audio_chunks}")
                        print(f"📊 Total audio bytes received: {total_audio_bytes:,}")
                        if audio_chunk_sizes:
                            print(f"📊 Audio chunk size stats:")
                            print(f"   Smallest: {min(audio_chunk_sizes):,} bytes")
                            print(f"   Largest: {max(audio_chunk_sizes):,} bytes")
                            print(f"   Average: {sum(audio_chunk_sizes) / len(audio_chunk_sizes):.0f} bytes")
                        print("=" * 60)
                        break
                        
                except json.JSONDecodeError as e:
                    print(f"⚠️ JSON decode error: {e}")
                    continue
                except Exception as e:
                    print(f"❌ Message processing error: {e}")
                    break
            
            # Wait for playback to complete
            await playback_task
            
            # Save collected MP3 chunks to file
            if self.mp3_chunks:
                print(f"💾 Saving {len(self.mp3_chunks)} audio chunks to {self.OUTPUT_FILE}...")
                await self._save_mp3_chunks_to_file(self.mp3_chunks)
            
            # Calculate streaming metrics
            streaming_end = time.time()
            total_streaming_time = streaming_end - streaming_start
            
            print("\n" + "=" * 60)
            print("�� PROGRESSIVE STREAMING METRICS:")
            print("=" * 60)
            print(f"Total streaming time: {total_streaming_time:.3f}s")
            print(f"Time to first audio: {time_to_first_audio:.3f}s")
            print(f"Total audio chunks: {total_audio_chunks}")
            print(f"Total audio bytes: {total_audio_bytes}")
            print(f"Average chunk size: {total_audio_bytes/total_audio_chunks:.0f} bytes")
            print(f"Streaming rate: {total_audio_bytes/total_streaming_time:.0f} bytes/s")
            print(f"Buffer size used: {self.BUFFER_SIZE} chunks")
            print(f"Output file: {self.OUTPUT_FILE}")
            print("=" * 60)
            
            # Cleanup
            stream.stop_stream()
            stream.close()
            
        except Exception as e:
            print(f"❌ Progressive streaming error: {e}")
            import traceback
            traceback.print_exc()
    
    async def _playback_worker(self, stream):
        """Background worker for smooth playback from buffer"""
        try:
            while not self.streaming_complete or len(self.audio_buffer) > 0:
                # Wait for playback to start
                if not self.playback_started:
                    await asyncio.sleep(0.1)
                    continue
                
                # Get audio chunk from buffer
                audio_data = None
                with self.buffer_lock:
                    if len(self.audio_buffer) > 0:
                        audio_data = self.audio_buffer.popleft()
                
                if audio_data:
                    # Play audio chunk
                    await self._play_audio_chunk(audio_data, stream)
                    
                    # Adaptive buffering - slow down if buffer is getting low
                    buffer_size = len(self.audio_buffer)
                    if buffer_size < self.MIN_BUFFER_SIZE:
                        await asyncio.sleep(0.05)  # Slow down playback
                    else:
                        await asyncio.sleep(0.01)  # Normal playback speed
                else:
                    await asyncio.sleep(0.01)
            
            print("🎵 Playback worker completed")
            
        except Exception as e:
            print(f"❌ Playback worker error: {e}")
    
    async def _play_audio_chunk(self, mp3_data, stream):
        """Play a single audio chunk"""
        try:
            from pydub import AudioSegment
            import io
            
            # Validate MP3 data
            if len(mp3_data) < 100:
                return
            
            # Decode MP3 to PCM with better error handling
            try:
            audio_segment = AudioSegment.from_file(io.BytesIO(mp3_data), format="mp3")
            except Exception as decode_error:
                # Try alternative decoding method
                try:
                    import tempfile
                    with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as temp_file:
                        temp_file.write(mp3_data)
                        temp_file.flush()
                        audio_segment = AudioSegment.from_file(temp_file.name, format="mp3")
                    import os
                    os.unlink(temp_file.name)
                except Exception as alt_error:
                    print(f"⚠️ Audio chunk decode failed: {decode_error}, alt: {alt_error}")
                    return
            
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
            print(f"⚠️ Audio chunk playback error: {e}")
            # Don't print full traceback for audio errors to avoid spam
    
    async def _save_mp3_chunks_to_file(self, mp3_chunks):
        """Save collected MP3 chunks to a single MP3 file with improved combining"""
        try:
            from pydub import AudioSegment
            import io
            
            print(f"�� Combining {len(mp3_chunks)} MP3 chunks with improved method...")
            
            # Convert all MP3 chunks to PCM first for better combining
            pcm_chunks = []
            
            for i, mp3_chunk in enumerate(mp3_chunks):
                try:
                    # Create audio segment from MP3 chunk
                    chunk_audio = AudioSegment.from_file(io.BytesIO(mp3_chunk), format="mp3")
                    
                    # Convert to consistent format (44.1kHz, mono)
                    if chunk_audio.frame_rate != 44100:
                        chunk_audio = chunk_audio.set_frame_rate(44100)
                    if chunk_audio.channels != 1:
                        chunk_audio = chunk_audio.set_channels(1)
                    
                    pcm_chunks.append(chunk_audio)
                    
                    if (i + 1) % 10 == 0:  # Progress update every 10 chunks
                        print(f"🔧 Processed {i + 1}/{len(mp3_chunks)} chunks...")
                        
                except Exception as e:
                    print(f"⚠️ Error processing chunk {i + 1}: {e}")
                    continue
            
            if pcm_chunks:
                print(f"�� Combining {len(pcm_chunks)} audio segments...")
                
                # Combine with crossfading to eliminate clicks
                combined_audio = pcm_chunks[0]
                
                for i in range(1, len(pcm_chunks)):
                    # Add small crossfade between chunks (50ms)
                    crossfade_duration = 50  # milliseconds
                    combined_audio = combined_audio.append(pcm_chunks[i], crossfade=crossfade_duration)
                    
                    if (i + 1) % 5 == 0:  # Progress update every 5 combinations
                        print(f"�� Combined {i + 1}/{len(pcm_chunks)} segments...")
                
                # Apply gentle fade in/out to eliminate edge clicks
                fade_duration = 10  # milliseconds
                combined_audio = combined_audio.fade_in(fade_duration).fade_out(fade_duration)
                
                # Export to MP3 file with high quality settings
                print(f"�� Exporting to {self.OUTPUT_FILE}...")
                combined_audio.export(
                    self.OUTPUT_FILE, 
                    format="mp3",
                    bitrate="192k",  # Higher bitrate for better quality
                    parameters=["-q:a", "0"]  # Highest quality setting
                )
                
                # Get file info
                file_size = os.path.getsize(self.OUTPUT_FILE)
                duration = len(combined_audio) / 1000.0  # Convert to seconds
                
                print(f"✅ Successfully saved to {self.OUTPUT_FILE}")
                print(f"📊 File size: {file_size:,} bytes")
                print(f"⏱️  Duration: {duration:.2f} seconds")
                print(f"🎵 Audio quality: {combined_audio.frame_rate}Hz, {combined_audio.channels} channel(s)")
                print(f"🔧 Applied crossfading and fade effects to eliminate clicks")
            else:
                print("❌ No valid audio chunks to save")
                
        except Exception as e:
            print(f"❌ Error saving MP3 file: {e}")
            import traceback
            traceback.print_exc()

async def main():
    """Main test function"""
    test = ProgressiveWebSocketStreamingTest()
    await test.stream_text_to_speech_progressive()

if __name__ == "__main__":
    asyncio.run(main())