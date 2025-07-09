#!/usr/bin/env python3
"""
Pro Speech-to-Speech Voice Changer
Leverages ElevenLabs Pro features for optimal streaming performance
"""

import asyncio
import json
import logging
import os
import sys
import time
from collections import deque
from typing import Optional, Dict, Any
import wave
import numpy as np
import pyaudio
import websockets
import aiohttp
from elevenlabs import generate, stream, set_api_key
from elevenlabs.api import History
import threading
import queue
import io
from pydub import AudioSegment

# Load environment variables from .env file if it exists
def load_env():
    """Load environment variables from .env file"""
    env_file = '.env'
    if os.path.exists(env_file):
        with open(env_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key.strip()] = value.strip()

# Load .env file
load_env()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pro_sts_voice_changer.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ProSTSConfig:
    """Pro-optimized configuration for Speech-to-Speech"""
    
    def __init__(self):
        # Audio settings optimized for Pro
        self.SAMPLE_RATE = 44100  # Pro supports 44.1kHz
        self.CHUNK_SIZE = 1024
        self.CHANNELS = 1
        self.FORMAT = pyaudio.paInt16
        
        # ElevenLabs Pro settings
        self.ELEVENLABS_API_KEY = os.getenv('ELEVENLABS_API_KEY')
        self.VOICE_ID = os.getenv('VOICE_ID', "pNInz6obpgDQGcFmaJgB")  # Adam voice
        self.MODEL_ID = "eleven_multilingual_sts_v2"  # Pro STS model
        
        # Pro-optimized streaming settings
        self.STREAM_CHUNK_SIZE = 1024
        self.MAX_CONCURRENT_REQUESTS = 10  # Pro limit
        self.AUDIO_QUALITY = 192  # Pro supports 192 kbps
        self.OUTPUT_FORMAT = "mp3_44100_192"  # Pro MP3 output
        
        # STS-specific settings
        self.STS_ENDPOINT = "https://api.elevenlabs.io/v1/speech-to-speech"
        self.STS_STREAM_ENDPOINT = "https://api.elevenlabs.io/v1/speech-to-speech/{voice_id}/stream"
        
        # Voice cloning settings (Pro feature)
        self.ENABLE_VOICE_CLONING = True
        self.CLONE_VOICE_NAME = "My Voice Clone"
        
        # Performance settings
        self.BUFFER_SIZE = 8192
        self.SILENCE_THRESHOLD = 0.01
        self.MIN_PHRASE_DURATION = 0.5
        self.MAX_PHRASE_DURATION = 10.0
        
        # Real-time settings
        self.LATENCY_TARGET = 0.1  # 100ms target latency
        self.PRIORITY_PROCESSING = True  # Pro feature
        
        # Phrase detection settings
        self.SILENCE_DURATION_THRESHOLD = 0.5  # Seconds of silence to end phrase
        self.MIN_AUDIO_LEVEL = 0.01  # Minimum audio level to consider speech (normalized like smooth script)

class ProSTSVoiceChanger:
    """Pro-optimized Speech-to-Speech voice changer"""
    
    def __init__(self, config: ProSTSConfig):
        self.config = config
        self.pyaudio = pyaudio.PyAudio()
        self.audio_queue = queue.Queue(maxsize=100)
        self.output_queue = queue.Queue(maxsize=100)
        self.is_running = False
        self.session = None
        self.websocket = None
        
        # Pro features
        self.concurrent_requests = 0
        self.request_semaphore = asyncio.Semaphore(config.MAX_CONCURRENT_REQUESTS)
        
        # Audio buffers
        self.input_buffer = deque(maxlen=int(config.SAMPLE_RATE * 2))  # 2 seconds
        self.output_buffer = deque(maxlen=int(config.SAMPLE_RATE * 2))
        
        # Performance tracking
        self.latency_history = deque(maxlen=100)
        self.throughput_history = deque(maxlen=100)
        
        # Phrase detection
        self.current_phrase = []
        self.phrase_start_time = None
        self.last_audio_time = None
        self.silence_start_time = None
        
        # Initialize ElevenLabs
        if config.ELEVENLABS_API_KEY:
            set_api_key(config.ELEVENLABS_API_KEY)
        else:
            raise ValueError("ELEVENLABS_API_KEY environment variable is required")
    
    async def initialize_pro_features(self):
        """Initialize Pro-specific features"""
        logger.info("Initializing Pro features...")
        
        # Test Pro capabilities
        await self.test_pro_capabilities()
        
        # Initialize voice cloning if enabled
        if self.config.ENABLE_VOICE_CLONING:
            await self.initialize_voice_cloning()
        
        # Setup WebSocket connection for real-time streaming
        await self.setup_websocket_connection()
        
        logger.info("Pro features initialized successfully")
    
    async def test_pro_capabilities(self):
        """Test Pro subscription capabilities"""
        logger.info("Testing Pro capabilities...")
        
        try:
            # Test API key and subscription
            async with aiohttp.ClientSession() as session:
                headers = {
                    "xi-api-key": self.config.ELEVENLABS_API_KEY,
                    "Content-Type": "application/json"
                }
                
                # Test subscription info
                async with session.get(
                    "https://api.elevenlabs.io/v1/user/subscription",
                    headers=headers
                ) as response:
                    if response.status == 200:
                        sub_info = await response.json()
                        logger.info(f"Subscription: {sub_info.get('tier', 'Unknown')}")
                        logger.info(f"Character count: {sub_info.get('character_count', 0)}")
                        logger.info(f"Character limit: {sub_info.get('character_limit', 0)}")
                    else:
                        logger.warning("Could not verify subscription status")
                
                # Test available models
                async with session.get(
                    "https://api.elevenlabs.io/v1/models",
                    headers=headers
                ) as response:
                    if response.status == 200:
                        models = await response.json()
                        turbo_models = [m for m in models if "turbo" in m.get("name", "").lower()]
                        logger.info(f"Available Turbo models: {len(turbo_models)}")
                        for model in turbo_models:
                            logger.info(f"  - {model['name']} ({model['model_id']})")
        
        except Exception as e:
            logger.error(f"Error testing Pro capabilities: {e}", exc_info=True)
    
    async def initialize_voice_cloning(self):
        """Initialize voice cloning (Pro feature)"""
        logger.info("Initializing voice cloning...")
        
        try:
            # Check if voice clone already exists
            async with aiohttp.ClientSession() as session:
                headers = {
                    "xi-api-key": self.config.ELEVENLABS_API_KEY,
                    "Content-Type": "application/json"
                }
                
                # List existing voices
                async with session.get(
                    "https://api.elevenlabs.io/v1/voices",
                    headers=headers
                ) as response:
                    if response.status == 200:
                        voices = await response.json()
                        existing_clone = next(
                            (v for v in voices.get("voices", []) 
                             if v.get("name") == self.config.CLONE_VOICE_NAME),
                            None
                        )
                        
                        if existing_clone:
                            logger.info(f"Found existing voice clone: {existing_clone['name']}")
                            self.config.VOICE_ID = existing_clone['voice_id']
                        else:
                            logger.info("No existing voice clone found. You can create one via the ElevenLabs dashboard.")
        
        except Exception as e:
            logger.error(f"Error initializing voice cloning: {e}", exc_info=True)
    
    async def setup_websocket_connection(self):
        """Setup WebSocket connection for real-time streaming"""
        logger.info("Setting up WebSocket connection...")
        
        try:
            # For now, use HTTP streaming (WebSocket support coming soon)
            logger.info("Using HTTP streaming (WebSocket not yet available)")
        
        except Exception as e:
            logger.error(f"Error setting up WebSocket: {e}", exc_info=True)
    
    def start_audio_capture(self):
        """Start real-time audio capture"""
        logger.info("Starting audio capture...")
        
        def audio_capture_thread():
            try:
                stream = self.pyaudio.open(
                    format=self.config.FORMAT,
                    channels=self.config.CHANNELS,
                    rate=self.config.SAMPLE_RATE,
                    input=True,
                    frames_per_buffer=self.config.CHUNK_SIZE,
                    stream_callback=self.audio_callback
                )
                
                stream.start_stream()
                
                while self.is_running:
                    time.sleep(0.001)
                
                stream.stop_stream()
                stream.close()
                
            except Exception as e:
                logger.error(f"Audio capture error: {e}", exc_info=True)
        
        self.audio_thread = threading.Thread(target=audio_capture_thread)
        self.audio_thread.start()
    
    def audio_callback(self, in_data, frame_count, time_info, status):
        """Audio callback for real-time processing"""
        try:
            # Convert audio data to numpy array
            audio_data = np.frombuffer(in_data, dtype=np.int16)
            
            # Add to input buffer
            self.input_buffer.extend(audio_data)
            
            # Check for silence using RMS (Root Mean Square) - normalized like smooth script
            audio_level = np.sqrt(np.mean(audio_data.astype(np.float32)**2)) / 32768.0
            current_time = time.time()
            
            # Debug: Log audio levels occasionally
            if hasattr(self, '_debug_counter'):
                self._debug_counter += 1
            else:
                self._debug_counter = 0
            
            if self._debug_counter % 100 == 0:  # Log every 100th callback
                logger.info(f"Audio callback called: level={audio_level:.6f}, threshold={self.config.MIN_AUDIO_LEVEL}")
            
            if audio_level > self.config.MIN_AUDIO_LEVEL:
                # Non-silent audio detected
                self.audio_queue.put({
                    'data': audio_data,
                    'timestamp': current_time,
                    'level': audio_level
                })
                self.last_audio_time = current_time
                self.silence_start_time = None
                logger.info(f"Audio detected: level={audio_level:.6f}")
            else:
                # Silence detected - match smooth script logic
                if self.silence_start_time is None:
                    self.silence_start_time = current_time
                    logger.info(f"Silence started at: {current_time}")
            
            return (in_data, pyaudio.paContinue)
        
        except Exception as e:
            logger.error(f"Audio callback error: {e}", exc_info=True)
            return (in_data, pyaudio.paContinue)
    
    async def process_audio_stream(self):
        """Process audio stream with Pro-optimized STS"""
        logger.info("Starting audio stream processing...")
        
        while self.is_running:
            try:
                # Get audio chunk with timeout
                audio_chunk = await asyncio.get_event_loop().run_in_executor(
                    None, lambda: self.audio_queue.get(timeout=0.1)
                )
                
                if audio_chunk is None:
                    continue
                
                # Add to current phrase
                self.current_phrase.append(audio_chunk['data'])
                
                # Start timing the phrase
                if self.phrase_start_time is None:
                    self.phrase_start_time = audio_chunk['timestamp']
                
                # Check phrase duration
                phrase_duration = audio_chunk['timestamp'] - self.phrase_start_time
                
                # Process phrase if it's long enough and we have silence
                if (phrase_duration >= self.config.MIN_PHRASE_DURATION and 
                    self.silence_start_time and 
                    audio_chunk['timestamp'] - self.silence_start_time >= self.config.SILENCE_DURATION_THRESHOLD):
                    
                    logger.info(f"Processing phrase: duration={phrase_duration:.2f}s, silence={(audio_chunk['timestamp'] - self.silence_start_time):.2f}s")
                    await self.process_phrase(self.current_phrase)
                    self.current_phrase = []
                    self.phrase_start_time = None
                    self.silence_start_time = None
                
                # Also process if max duration reached
                elif phrase_duration >= self.config.MAX_PHRASE_DURATION:
                    logger.info(f"Processing phrase due to max duration: {phrase_duration:.2f}s")
                    await self.process_phrase(self.current_phrase)
                    self.current_phrase = []
                    self.phrase_start_time = None
                    self.silence_start_time = None
                
                # Force process if phrase is getting too long (2+ seconds)
                elif phrase_duration >= 2.0 and len(self.current_phrase) > 30:
                    logger.info(f"Force processing long phrase: duration={phrase_duration:.2f}s, chunks={len(self.current_phrase)}")
                    await self.process_phrase(self.current_phrase)
                    self.current_phrase = []
                    self.phrase_start_time = None
                    self.silence_start_time = None
                
                # Debug: Log phrase state occasionally
                if hasattr(self, '_phrase_debug_counter'):
                    self._phrase_debug_counter += 1
                else:
                    self._phrase_debug_counter = 0
                
                if self._phrase_debug_counter % 50 == 0:  # Log every 50th iteration
                    silence_duration = (audio_chunk['timestamp'] - self.silence_start_time) if self.silence_start_time else 0
                    logger.info(f"Phrase state: chunks={len(self.current_phrase)}, duration={phrase_duration:.2f}s, silence={silence_duration:.2f}s")
            
            except queue.Empty:
                # Check for timeout-based processing
                if (self.current_phrase and self.phrase_start_time and 
                    self.silence_start_time and 
                    time.time() - self.silence_start_time >= self.config.SILENCE_DURATION_THRESHOLD):
                    
                    phrase_duration = time.time() - self.phrase_start_time
                    if phrase_duration >= self.config.MIN_PHRASE_DURATION:
                        logger.info(f"Processing phrase due to timeout: {phrase_duration:.2f}s")
                        await self.process_phrase(self.current_phrase)
                        self.current_phrase = []
                        self.phrase_start_time = None
                        self.silence_start_time = None
                
                # Also force process if phrase is getting too long without silence
                elif (self.current_phrase and self.phrase_start_time and 
                      time.time() - self.phrase_start_time >= 2.0 and 
                      len(self.current_phrase) > 20):
                    
                    phrase_duration = time.time() - self.phrase_start_time
                    logger.info(f"Force processing phrase due to timeout: {phrase_duration:.2f}s, chunks={len(self.current_phrase)}")
                    await self.process_phrase(self.current_phrase)
                    self.current_phrase = []
                    self.phrase_start_time = None
                    self.silence_start_time = None
                continue
            
            except Exception as e:
                logger.error(f"Error processing audio stream: {e}", exc_info=True)
    
    async def process_phrase(self, audio_chunks):
        """Process a complete phrase with Pro STS"""
        if not audio_chunks:
            return
        
        logger.info(f"Processing phrase with {len(audio_chunks)} chunks")
        
        try:
            # Combine audio chunks
            combined_audio = np.concatenate(audio_chunks)
            
            # Debug: Check combined audio
            logger.info(f"Combined audio shape: {combined_audio.shape}, total samples: {len(combined_audio)}")
            
            # Convert to proper format for ElevenLabs
            audio_bytes = combined_audio.tobytes()
            
            # Process with Pro STS
            start_time = time.time()
            
            async with self.request_semaphore:
                self.concurrent_requests += 1
                
                try:
                    # Use Pro STS streaming
                    output_audio = await self.stream_sts_pro(audio_bytes)
                    
                    if output_audio:
                        # Calculate latency
                        latency = time.time() - start_time
                        self.latency_history.append(latency)
                        
                        # Add to output queue
                        await asyncio.get_event_loop().run_in_executor(
                            None, self.output_queue.put, output_audio
                        )
                        
                        logger.info(f"STS processed phrase: {len(audio_chunks)} chunks, "
                                  f"latency: {latency:.3f}s")
                    else:
                        logger.warning("No output audio received from STS")
                
                finally:
                    self.concurrent_requests -= 1
        
        except Exception as e:
            logger.error(f"Error processing phrase: {e}", exc_info=True)
    
    async def stream_sts_pro(self, audio_data: bytes) -> Optional[bytes]:
        """Stream audio through Pro STS with optimized settings"""
        try:
            # Convert numpy array to WAV format
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
            
            # Debug: Check audio data
            logger.info(f"Audio array shape: {audio_array.shape}, dtype: {audio_array.dtype}")
            logger.info(f"Audio array min/max: {audio_array.min()}/{audio_array.max()}")
            
            # Create WAV file in memory
            wav_buffer = io.BytesIO()
            
            with wave.open(wav_buffer, 'wb') as wav_file:
                wav_file.setnchannels(self.config.CHANNELS)
                wav_file.setsampwidth(2)  # 16-bit = 2 bytes
                wav_file.setframerate(self.config.SAMPLE_RATE)
                wav_file.writeframes(audio_array.tobytes())
            
            wav_data = wav_buffer.getvalue()
            logger.info(f"WAV data size: {len(wav_data)} bytes")
            
            headers = {
                "xi-api-key": self.config.ELEVENLABS_API_KEY,
                "Accept": "audio/mpeg"
            }
            
            # Prepare request data
            data = {
                "model_id": self.config.MODEL_ID,
                "output_format": self.config.OUTPUT_FORMAT,
                "audio_quality": self.config.AUDIO_QUALITY
            }
            
            # Debug: Log what we're sending
            logger.info(f"STS request data: voice_id={self.config.VOICE_ID} (in URL), model_id={self.config.MODEL_ID}")
            
            # Create multipart form data
            form_data = aiohttp.FormData()
            form_data.add_field('audio', wav_data, filename='input.wav', content_type='audio/wav')
            
            for key, value in data.items():
                form_data.add_field(key, str(value))
                logger.info(f"STS form field: {key}={value}")
            
            logger.info(f"Sending STS request with {len(wav_data)} bytes of audio data")
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.config.STS_STREAM_ENDPOINT.format(voice_id=self.config.VOICE_ID),
                    headers=headers,
                    data=form_data
                ) as response:
                    
                    if response.status == 200:
                        # Read streaming response
                        output_audio = b""
                        async for chunk in response.content.iter_chunked(1024):
                            output_audio += chunk
                        
                        logger.info(f"STS response: {len(output_audio)} bytes")
                        return output_audio
                    else:
                        error_text = await response.text()
                        logger.error(f"STS API error: {response.status} - {error_text}")
                        return None
        
        except Exception as e:
            logger.error(f"Error in Pro STS streaming: {e}", exc_info=True)
            return None
    
    def start_audio_output(self):
        """Start real-time audio output"""
        logger.info("Starting audio output...")
        
        def output_callback():
            try:
                stream = self.pyaudio.open(
                    format=pyaudio.paFloat32,  # Play as float32 for smooth playback
                    channels=self.config.CHANNELS,
                    rate=self.config.SAMPLE_RATE,
                    output=True,
                    frames_per_buffer=self.config.CHUNK_SIZE
                )
                
                while self.is_running:
                    try:
                        # Get output audio
                        output_audio = self.output_queue.get(timeout=0.1)
                        
                        if output_audio:
                            # Decode MP3 to float32 PCM using pydub
                            try:
                                audio_segment = AudioSegment.from_file(io.BytesIO(output_audio), format="mp3")
                                if len(audio_segment) == 0:
                                    continue
                                pcm_data = audio_segment.get_array_of_samples()
                                pcm_float = np.array(pcm_data, dtype=np.float32) / 32768.0
                                stream.write(pcm_float.astype(np.float32).tobytes())
                            except Exception as e:
                                logger.error(f"Audio decode/playback error: {e}", exc_info=True)
                                continue
                        
                    except queue.Empty:
                        continue
                    except Exception as e:
                        logger.error(f"Audio output error: {e}", exc_info=True)
                
                stream.stop_stream()
                stream.close()
                
            except Exception as e:
                logger.error(f"Audio output thread error: {e}", exc_info=True)
        
        self.output_thread = threading.Thread(target=output_callback)
        self.output_thread.start()
    
    async def run(self):
        """Run the Pro STS voice changer"""
        logger.info("Starting Pro STS Voice Changer...")
        
        try:
            # Initialize Pro features
            await self.initialize_pro_features()
            
            # Start audio capture and output
            self.is_running = True
            self.start_audio_capture()
            self.start_audio_output()
            
            # Start audio processing
            await self.process_audio_stream()
        
        except KeyboardInterrupt:
            logger.info("Shutting down...")
        except Exception as e:
            logger.error(f"Error in main loop: {e}", exc_info=True)
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Cleanup resources"""
        logger.info("Cleaning up...")
        
        self.is_running = False
        
        if hasattr(self, 'audio_thread'):
            self.audio_thread.join(timeout=1)
        
        if hasattr(self, 'output_thread'):
            self.output_thread.join(timeout=1)
        
        if self.pyaudio:
            self.pyaudio.terminate()
        
        logger.info("Cleanup complete")

async def main():
    """Main function"""
    config = ProSTSConfig()
    voice_changer = ProSTSVoiceChanger(config)
    
    try:
        await voice_changer.run()
    except Exception as e:
        logger.error(f"Main error: {e}", exc_info=True)

if __name__ == "__main__":
    asyncio.run(main()) 