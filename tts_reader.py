#!/usr/bin/env python3
"""
TTS Reader - Reads from text file and streams to ElevenLabs
Similar to websocket_streaming_test.py but monitors a file for changes
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

class TTSReader:
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
        
        # File settings
        self.TEXT_FILE = "voice_changer_text.txt"
        self.last_file_size = 0
        self.last_processed_text = ""
        
        # Control
        self.running = True
    
    async def connect_websocket(self):
        """Connect to ElevenLabs WebSocket"""
        try:
            uri = (
                f"wss://api.elevenlabs.io/v1/text-to-speech/{self.VOICE_ID}/stream-input"
                f"?model_id={self.MODEL_ID}&optimize_streaming_latency=4"
            )
            headers = {
                "xi-api-key": self.ELEVENLABS_API_KEY,
                "Content-Type": "application/json"
            }
            
            print(f"🔗 Connecting to ElevenLabs WebSocket...")
            websocket = await websockets.connect(uri, extra_headers=headers)
            print("✅ Connected to ElevenLabs WebSocket")
            
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
            
            return websocket
            
        except Exception as e:
            print(f"❌ Failed to connect to ElevenLabs WebSocket: {e}")
            return None
    
    def read_text_file(self):
        """Read text from file and return the first complete phrase"""
        try:
            if not os.path.exists(self.TEXT_FILE):
                return ""
            
            with open(self.TEXT_FILE, 'r', encoding='utf-8') as f:
                content = f.read().strip()
            
            # Return the first line (first complete phrase)
            if content:
                first_line = content.split('\n')[0].strip()
                return first_line
            
            return ""
        except Exception as e:
            print(f"❌ Error reading text file: {e}")
            return ""
    
    def remove_processed_text(self, processed_text):
        """Remove only the processed text from the file, keeping any remaining text"""
        try:
            if not os.path.exists(self.TEXT_FILE):
                return
            
            with open(self.TEXT_FILE, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Remove the processed text and its newline from the beginning
            text_to_remove = processed_text + "\n"
            if content.startswith(text_to_remove):
                remaining_content = content[len(text_to_remove):]
                
                with open(self.TEXT_FILE, 'w', encoding='utf-8') as f:
                    f.write(remaining_content)
                
                print(f"🗑️  Removed processed text: '{processed_text}'")
                if remaining_content.strip():
                    print(f"📄 Remaining phrases: {remaining_content.strip()}")
                else:
                    print(f"📄 No remaining text")
            else:
                print(f"⚠️  Could not find processed text at beginning of file")
                
        except Exception as e:
            print(f"❌ Error removing processed text: {e}")
    
    def clear_text_file(self):
        """Clear the text file after processing"""
        try:
            with open(self.TEXT_FILE, 'w', encoding='utf-8') as f:
                f.write("")
            print("🗑️  Cleared text file")
        except Exception as e:
            print(f"❌ Error clearing text file: {e}")
    
    def cleanup_corrupted_file(self):
        """Clean up corrupted or duplicate content in the file"""
        try:
            if not os.path.exists(self.TEXT_FILE):
                return
            
            with open(self.TEXT_FILE, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Split into lines and remove duplicates
            lines = content.split('\n')
            unique_lines = []
            seen = set()
            
            for line in lines:
                line = line.strip()
                if line and line not in seen:
                    unique_lines.append(line)
                    seen.add(line)
            
            # Write back cleaned content
            with open(self.TEXT_FILE, 'w', encoding='utf-8') as f:
                f.write('\n'.join(unique_lines))
            
            print(f"🧹 Cleaned up file, removed duplicates")
            
        except Exception as e:
            print(f"❌ Error cleaning up file: {e}")
            # If cleanup fails, just clear the file
            self.clear_text_file()
    
    async def stream_text_and_play_audio(self, websocket, text):
        """Stream text to ElevenLabs and play audio immediately"""
        try:
            print(f"🎵 Streaming text: '{text}'")
            
            # Send text
            text_message = {
                "text": text,
                "try_trigger_generation": True
            }
            await websocket.send(json.dumps(text_message))
            
            # Send end marker
            await websocket.send(json.dumps({"text": ""}))
            
            # Initialize audio output
            stream = self.audio.open(
                format=self.FORMAT,
                channels=self.CHANNELS,
                rate=self.SAMPLE_RATE,
                output=True,
                frames_per_buffer=self.CHUNK_SIZE
            )
            
            print("🎵 Audio output stream initialized")
            
            # Receive and play audio
            async for message in websocket:
                try:
                    data = json.loads(message)
                    
                    if "audio" in data and data["audio"]:
                        # Decode and play audio
                        audio_data = base64.b64decode(data["audio"])
                        await self._play_audio_chunk(audio_data, stream)
                        print(f"🔊 Audio chunk: {len(audio_data)} bytes")
                    
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
            
            # Cleanup audio stream
            stream.stop_stream()
            stream.close()
            
        except Exception as e:
            print(f"❌ Error in stream_text_and_play_audio: {e}")
    
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
    
    async def run(self):
        """Main TTS reader loop"""
        print("🎵 TTS Reader: Starting...")
        print("📁 Monitoring file:", self.TEXT_FILE)
        print("=" * 50)
        
        # Create empty text file if it doesn't exist
        if not os.path.exists(self.TEXT_FILE):
            with open(self.TEXT_FILE, 'w', encoding='utf-8') as f:
                f.write("")
            print("📄 Created empty text file")
        
        while self.running:
            try:
                # Connect to WebSocket for each text stream
                websocket = await self.connect_websocket()
                if not websocket:
                    await asyncio.sleep(1)
                    continue
                
                # Check if file has changed
                current_size = os.path.getsize(self.TEXT_FILE) if os.path.exists(self.TEXT_FILE) else 0
                
                # If file is getting too large, clean it up
                if current_size > 10000:  # More than 10KB
                    print("⚠️  File is getting large, cleaning up...")
                    self.cleanup_corrupted_file()
                    current_size = os.path.getsize(self.TEXT_FILE) if os.path.exists(self.TEXT_FILE) else 0
                
                if current_size != self.last_file_size:
                    # Read new text
                    current_text = self.read_text_file()
                    
                    if current_text and current_text != self.last_processed_text:
                        print(f"📖 New text detected: '{current_text}'")
                        
                        try:
                            # Stream text and play audio
                            await self.stream_text_and_play_audio(websocket, current_text)
                            
                            # Update tracking
                            self.last_processed_text = current_text
                            
                            # Remove only the processed text, keep any remaining text
                            self.remove_processed_text(current_text)
                            self.last_file_size = os.path.getsize(self.TEXT_FILE) if os.path.exists(self.TEXT_FILE) else 0
                            
                        except Exception as e:
                            print(f"❌ Error processing text '{current_text}': {e}")
                            # Remove the problematic text to prevent endless loops
                            self.remove_processed_text(current_text)
                            self.last_file_size = os.path.getsize(self.TEXT_FILE) if os.path.exists(self.TEXT_FILE) else 0
                            await asyncio.sleep(1)  # Wait before trying again
                    else:
                        self.last_file_size = current_size
                
                # Close WebSocket after each stream to avoid timeout
                await websocket.close()
                
                await asyncio.sleep(0.1)  # Check every 100ms
                
            except Exception as e:
                print(f"❌ Error in main loop: {e}")
                await asyncio.sleep(1)
        
        # Cleanup
        self.audio.terminate()
        print("✅ TTS Reader stopped")

async def main():
    """Main function"""
    # Check required environment variables
    if not os.getenv("ELEVENLABS_API_KEY"):
        print("❌ ELEVENLABS_API_KEY not found in environment variables")
        print("   Please add it to your .env file")
        return
    
    print("✅ Prerequisites check passed")
    
    # Create and run TTS reader
    tts_reader = TTSReader()
    
    try:
        await tts_reader.run()
    except KeyboardInterrupt:
        print("\n⏹️  Stopping TTS reader...")
        tts_reader.running = False
        print("👋 TTS reader stopped")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    asyncio.run(main()) 