#!/usr/bin/env python3
"""
WebSocket Streaming Test for ElevenLabs TTS
Test continuous streaming with longer Russian text
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

# Load environment variables
load_dotenv()

class WebSocketStreamingTest:
    def __init__(self):
        self.ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
        self.VOICE_ID = os.getenv("VOICE_ID", "21m00Tcm4TlvDq8ikWAM")
        self.MODEL_ID = "eleven_multilingual_v2"
        
        # Audio settings
        self.SAMPLE_RATE = 44100
        self.CHANNELS = 1
        self.FORMAT = pyaudio.paFloat32
        self.CHUNK_SIZE = 1024
        
        # PyAudio instance
        self.audio = pyaudio.PyAudio()
        
        # Output file settings
        self.OUTPUT_FILE = "websocket_streaming_output.mp3"
        
        # Test text (longer Russian text)
        self.test_text = """
Так почему же вы здесь?
     - А где же мне быть? Где же мне работать, по-твоему? В школе? Что я там
буду воровать, промокашки?!  Устраиваясь на работу, ты  должен  прежде всего
задуматься: что, где и как? Что я смогу украсть? Где  я смогу украсть? И как
я  смогу украсть?.. Ты  понял?  Вот и хорошо. Все будет нормально.  К вечеру
бабки появятся.
"""
    
    async def stream_text_to_speech(self):
        """Stream text to speech using ElevenLabs WebSocket"""
        if not self.ELEVENLABS_API_KEY:
            print("❌ ELEVENLABS_API_KEY not found")
            return
        
        print("🎵 Starting WebSocket Streaming Test...")
        print("=" * 60)
        print("Text to stream:")
        print(self.test_text[:100] + "..." if len(self.test_text) > 100 else self.test_text)
        print("=" * 60)
        
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
                
                for i, chunk in enumerate(text_chunks):
                    text_message = {
                        "text": chunk,
                        "try_trigger_generation": True
                    }
                    await websocket.send(json.dumps(text_message))
                    print(f"📤 Sent chunk {i+1}/{len(text_chunks)}: '{chunk[:30]}...'")
                    await asyncio.sleep(0.1)  # Small delay between chunks
                
                # Send end marker
                await websocket.send(json.dumps({"text": ""}))
                print("📤 Sent end marker")
                
                # Start audio playback
                print("🔊 Starting audio playback...")
                await self._play_streaming_audio(websocket)
                
        except Exception as e:
            print(f"❌ WebSocket streaming error: {e}")
            import traceback
            traceback.print_exc()
    
    def _split_text_into_chunks(self, text, max_chunk_size=100):
        """Split text into chunks for streaming"""
        words = text.split()
        chunks = []
        current_chunk = ""
        
        for word in words:
            if len(current_chunk + " " + word) <= max_chunk_size:
                current_chunk += (" " + word) if current_chunk else word
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = word
        
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks
    
    async def _play_streaming_audio(self, websocket):
        """Play audio as it streams from WebSocket and save to MP3 file"""
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
            
            # Collect MP3 chunks for file saving
            mp3_chunks = []
            
            # Receive and play audio chunks
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
                        
                        # Collect for file saving
                        mp3_chunks.append(audio_data)
                        
                        # Play audio immediately
                        await self._play_audio_chunk(audio_data, stream)
                        
                        print(f"🔊 Audio chunk {total_audio_chunks}: {len(audio_data)} bytes")
                    
                    elif "audio" in data and data["audio"] is None:
                        print("📡 End of stream signal received")
                    
                    if data.get("isFinal"):
                        print("✅ Stream completed")
                        break
                        
                except json.JSONDecodeError as e:
                    print(f"⚠️ JSON decode error: {e}")
                    continue
                except Exception as e:
                    print(f"❌ Message processing error: {e}")
                    break
            
            # Save collected MP3 chunks to file
            if mp3_chunks:
                print(f"💾 Saving {len(mp3_chunks)} audio chunks to {self.OUTPUT_FILE}...")
                await self._save_mp3_chunks_to_file(mp3_chunks)
            
            # Calculate streaming metrics
            streaming_end = time.time()
            total_streaming_time = streaming_end - streaming_start
            
            print("\n" + "=" * 60)
            print("📊 STREAMING METRICS:")
            print("=" * 60)
            print(f"Total streaming time: {total_streaming_time:.3f}s")
            print(f"Time to first audio: {time_to_first_audio:.3f}s")
            print(f"Total audio chunks: {total_audio_chunks}")
            print(f"Total audio bytes: {total_audio_bytes}")
            print(f"Average chunk size: {total_audio_bytes/total_audio_chunks:.0f} bytes")
            print(f"Streaming rate: {total_audio_bytes/total_streaming_time:.0f} bytes/s")
            print(f"Output file: {self.OUTPUT_FILE}")
            print("=" * 60)
            
            # Cleanup
            stream.stop_stream()
            stream.close()
            
        except Exception as e:
            print(f"❌ Audio playback error: {e}")
            import traceback
            traceback.print_exc()
    
    async def _play_audio_chunk(self, mp3_data, stream):
        """Play a single audio chunk"""
        try:
            from pydub import AudioSegment
            import io
            
            # Validate MP3 data
            if len(mp3_data) < 100:
                return
            
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
            print(f"⚠️ Audio chunk playback error: {e}")
    
    async def _save_mp3_chunks_to_file(self, mp3_chunks):
        """Save collected MP3 chunks to a single MP3 file with improved combining"""
        try:
            from pydub import AudioSegment
            import io
            
            print(f"🔧 Combining {len(mp3_chunks)} MP3 chunks with improved method...")
            
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
                print(f"🔧 Combining {len(pcm_chunks)} audio segments...")
                
                # Combine with crossfading to eliminate clicks
                combined_audio = pcm_chunks[0]
                
                for i in range(1, len(pcm_chunks)):
                    # Add small crossfade between chunks (50ms)
                    crossfade_duration = 50  # milliseconds
                    combined_audio = combined_audio.append(pcm_chunks[i], crossfade=crossfade_duration)
                    
                    if (i + 1) % 5 == 0:  # Progress update every 5 combinations
                        print(f"🔧 Combined {i + 1}/{len(pcm_chunks)} segments...")
                
                # Apply gentle fade in/out to eliminate edge clicks
                fade_duration = 10  # milliseconds
                combined_audio = combined_audio.fade_in(fade_duration).fade_out(fade_duration)
                
                # Export to MP3 file with high quality settings
                print(f"💾 Exporting to {self.OUTPUT_FILE}...")
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
    test = WebSocketStreamingTest()
    await test.stream_text_to_speech()

if __name__ == "__main__":
    asyncio.run(main()) 