#!/usr/bin/env python3
"""
Real-time Voice Changer for Windows - Optimized for Low Latency
Uses ElevenLabs Pro with eleven_flash_v2_5 model and Google credentials
Works with webrtcvad-wheels for VAD
"""

import asyncio
import json
import logging
import os
import queue
import threading
import time
from typing import Optional, Dict, Any

import numpy as np
import sounddevice as sd
import soundfile as sf
from dotenv import load_dotenv
import openai
from elevenlabs import generate, stream, set_api_key
import requests
import tempfile
from pydub import AudioSegment

# Try to import webrtcvad-wheels first, fallback to webrtcvad
try:
    import webrtcvad_wheels as webrtcvad
    print("✓ Using webrtcvad-wheels for VAD")
except ImportError:
    try:
        import webrtcvad
        print("✓ Using webrtcvad for VAD")
    except ImportError:
        print("✗ No VAD library found. Please install webrtcvad-wheels")
        exit(1)

# Load environment variables
load_dotenv()

# Configure logging
# Set logger to DEBUG level
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RealTimeVoiceChanger:
    def __init__(self):
        # Audio settings - optimized for lower latency
        self.sample_rate = 16000
        self.chunk_duration = 0.0625  # seconds (reduced from 0.125 for lower latency)
        self.chunk_size = int(self.sample_rate * self.chunk_duration)
        self.channels = 1
        
        # Voice settings
        self.target_voice_id = os.getenv('ELEVENLABS_VOICE_ID', 'GN4wbsbejSnGSa1AzjH5')  # Read from env, fallback to Ekaterina
        self.elevenlabs_model = "eleven_flash_v2_5"
        
        # VAD settings
        self.vad = webrtcvad.Vad(1)  # Reduced aggressiveness for better detection
        self.frame_duration = 30  # ms
        
        # Queues for audio processing
        self.input_queue = queue.Queue(maxsize=100)  # Increased queue size
        self.output_queue = queue.Queue(maxsize=100)  # Increased queue size
        self.text_queue = queue.Queue(maxsize=20)  # Increased queue size
        
        # State management - optimized for faster processing
        self.is_running = False
        self.is_speaking = False
        self.speech_buffer = []
        self.silence_frames = 0
        self.silence_threshold = 8  # frames of silence before processing (reduced from 15)
        self.force_transcription_counter = 0  # For debug: count chunks
        self.force_transcription_limit = 12   # Force transcription after 12 chunks (reduced from 20)
        self.max_speech_buffer_chunks = 8     # Max speech buffer length (0.5 seconds at 0.0625s chunks)
        
        # Initialize APIs
        self._setup_apis()
        
        self.output_audio_buffer = np.array([], dtype=np.float32)  # Persistent output buffer
    
    def _setup_apis(self):
        """Setup API keys and clients"""
        # ElevenLabs
        elevenlabs_key = os.getenv('ELEVENLABS_API_KEY')
        if not elevenlabs_key:
            raise ValueError("ELEVENLABS_API_KEY not found in environment variables")
        set_api_key(elevenlabs_key)
        
        # Validate voice ID
        if not self.target_voice_id:
            raise ValueError("ELEVENLABS_VOICE_ID not found in environment variables")
        logger.info(f"Using ElevenLabs voice ID: {self.target_voice_id}")
        
        # OpenAI (for transcription)
        openai_key = os.getenv('OPENAI_API_KEY')
        if not openai_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        openai.api_key = openai_key
        
        logger.info("APIs configured successfully")
    
    def _audio_callback(self, indata, frames, time, status):
        """Callback for audio input"""
        if status:
            logger.warning(f"Audio input status: {status}")
        
        # Convert to mono if stereo
        if indata.shape[1] > 1:
            audio_data = np.mean(indata, axis=1)
        else:
            audio_data = indata.flatten()
        
        # Resample if needed
        if len(audio_data) != self.chunk_size:
            audio_data = self._resample_audio(audio_data, self.chunk_size)
        
        # Add to input queue with non-blocking approach
        try:
            # Try to put without blocking
            self.input_queue.put_nowait(audio_data)
        except queue.Full:
            # If queue is full, try to clear some space by getting one item
            try:
                self.input_queue.get_nowait()
                self.input_queue.put_nowait(audio_data)
                logger.debug("Input queue was full, cleared one item and added new chunk")
            except queue.Empty:
                logger.warning("Input queue full, dropping audio chunk")
    
    def _resample_audio(self, audio_data, target_length):
        """Simple resampling by interpolation"""
        if len(audio_data) == target_length:
            return audio_data
        
        indices = np.linspace(0, len(audio_data) - 1, target_length)
        return np.interp(indices, np.arange(len(audio_data)), audio_data)
    
    def _detect_speech(self, audio_chunk):
        """Detect if audio chunk contains speech using VAD"""
        try:
            # Convert to int16 for VAD
            audio_int16 = (audio_chunk * 32767).astype(np.int16)
            frame_size = int(self.sample_rate * self.frame_duration / 1000)
            speech_detected = False
            logger.debug(f"VAD input: len={len(audio_int16)}, frame_size={frame_size}, dtype={audio_int16.dtype}, min={audio_int16.min()}, max={audio_int16.max()}")
            for i in range(0, len(audio_int16) - frame_size, frame_size):
                frame = audio_int16[i:i + frame_size]
                if len(frame) == frame_size:
                    is_speech = self.vad.is_speech(frame.tobytes(), self.sample_rate)
                    logger.debug(f"VAD frame {i//frame_size}: is_speech={is_speech}")
                    if is_speech:
                        speech_detected = True
                        break
            logger.debug(f"VAD result: {speech_detected}")
            return speech_detected
        except Exception as e:
            logger.error(f"VAD error: {e}")
            return True  # Default to speech if VAD fails
    
    async def _transcribe_audio(self, audio_chunk):
        """Transcribe audio using OpenAI Whisper (new API)"""
        try:
            # Log audio details for debugging
            audio_duration = len(audio_chunk) / self.sample_rate
            logger.info(f"Transcribing audio: {len(audio_chunk)} samples, {audio_duration:.2f}s duration")
            logger.debug(f"Audio stats: min={audio_chunk.min():.6f}, max={audio_chunk.max():.6f}, mean={audio_chunk.mean():.6f}")
            
            # Save audio to temporary file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                sf.write(temp_file, audio_chunk, self.sample_rate)
                temp_file_path = temp_file.name
                logger.debug(f"Saved audio to temp file: {temp_file_path}")
            
            # Prepare the request
            headers = {
                "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}"
            }
            
            logger.debug("Preparing Whisper API request...")
            with open(temp_file_path, "rb") as audio_file:
                files = {
                    "file": ("audio.wav", audio_file, "audio/wav"),
                    "model": (None, "whisper-1"),  # or "whisper-2"
                    "language": (None, "ru"),  # Set to Russian
                    "response_format": (None, "json"),
                    "temperature": (None, "0.0"),
                }
                
                logger.debug("Sending request to Whisper API...")
                response = requests.post(
                    "https://api.openai.com/v1/audio/transcriptions",
                    headers=headers,
                    files=files
                )
            
            os.unlink(temp_file_path)
            logger.debug(f"Whisper API response: status={response.status_code}, text={response.text[:200]}")
            
            if response.status_code == 200:
                result = response.json()
                text = result.get("text", "").strip()
                if text:
                    logger.info(f"Transcribed ({audio_duration:.2f}s): '{text}'")
                    return text
                else:
                    logger.warning(f"Whisper returned empty text for {audio_duration:.2f}s audio")
            else:
                logger.error(f"Whisper API error: {response.status_code} - {response.text}")
        except Exception as e:
            logger.error(f"Transcription error: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
        return None

    async def _generate_speech(self, text):
        """Generate speech using ElevenLabs"""
        try:
            logger.info(f"Generating speech for: '{text}'")
            
            # Use the generate function with proper parameters
            audio_stream = generate(
                text=text,
                voice=self.target_voice_id,
                model=self.elevenlabs_model,
                stream=True
            )
            
            # Collect audio chunks
            audio_chunks = []
            async def collect_chunks():
                try:
                    for chunk in audio_stream:
                        if chunk:
                            audio_chunks.append(chunk)
                    logger.debug(f"Collected {len(audio_chunks)} audio chunks from ElevenLabs")
                except Exception as e:
                    logger.error(f"Error collecting audio chunks: {e}")
            
            # Run the collection
            await collect_chunks()
            
            if audio_chunks:
                # Combine all chunks
                combined_audio = b''.join(audio_chunks)
                
                # Convert to numpy array
                audio_array = np.frombuffer(combined_audio, dtype=np.int16)
                # Convert to float32 and normalize
                audio_float = audio_array.astype(np.float32) / 32767.0
                
                # Resample from 44.1kHz to 16kHz if needed
                if len(audio_float) > 0:
                    # Simple downsampling by taking every 2.75th sample (44100/16000 ≈ 2.75)
                    downsample_factor = int(44100 / 16000)
                    audio_float = audio_float[::downsample_factor]
                    
                    logger.info(f"Generated speech: {len(audio_float)} samples ({len(audio_float)/16000:.2f}s)")
                    
                    # Save debug output
                    debug_output_path = f"debug_output_{int(time.time())}.wav"
                    try:
                        sf.write(debug_output_path, audio_float, 16000)
                        logger.debug(f"Saved debug output to {debug_output_path}")
                    except Exception as e:
                        logger.error(f"Failed to save debug output: {e}")
                    
                    return audio_float
                else:
                    logger.warning("No audio data generated")
            else:
                logger.warning("No audio chunks received from ElevenLabs")
        except Exception as e:
            logger.error(f"Speech generation error: {e}")
        return None

    async def _process_speech_buffer(self, combined_audio, reason=""):
        """Process speech buffer asynchronously"""
        try:
            logger.info(f"Starting speech buffer processing ({reason}) with {len(combined_audio)} samples")
            
            # Transcribe
            logger.debug("Calling transcription...")
            text = await self._transcribe_audio(combined_audio)
            logger.debug(f"Transcription result: {text}")
            
            if text:
                logger.info(f"Transcribed text ({reason}): {text}")
                # Generate speech
                logger.info(f"Calling TTS for ({reason}): {text}")
                output_audio = await self._generate_speech(text)
                if output_audio is not None:
                    try:
                        # Append to persistent output buffer
                        if self.output_audio_buffer.size == 0:
                            self.output_audio_buffer = output_audio
                        else:
                            self.output_audio_buffer = np.concatenate([self.output_audio_buffer, output_audio])
                        logger.debug(f"Appended {len(output_audio)} samples to output_audio_buffer. New length: {len(self.output_audio_buffer)}")
                    except Exception as e:
                        logger.warning(f"Failed to append TTS audio to output buffer: {e}")
                else:
                    logger.warning(f"Failed to generate speech for text: {text}")
            else:
                logger.warning(f"No text transcribed from speech buffer ({reason})")
        except Exception as e:
            logger.error(f"Error processing speech buffer: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")

    async def _audio_processor(self):
        """Main audio processing loop - optimized for lower latency"""
        logger.info("Starting audio processor...")
        
        while self.is_running:
            try:
                # Check if input queue is getting too full and clear some items
                if self.input_queue.qsize() > 80:  # If more than 80% full
                    logger.warning(f"Input queue getting full ({self.input_queue.qsize()}/100), clearing old items")
                    # Clear some old items
                    for _ in range(20):
                        try:
                            self.input_queue.get_nowait()
                        except queue.Empty:
                            break
                
                # Get audio chunk from input queue
                audio_chunk = self.input_queue.get(timeout=0.5)  # Reduced timeout
                logger.debug(f"Audio chunk received: shape={audio_chunk.shape}, dtype={audio_chunk.dtype}, min={audio_chunk.min()}, max={audio_chunk.max()}, mean={audio_chunk.mean()}")
                
                # Save raw audio for debugging
                debug_audio_path = f"debug_input_{int(time.time())}.wav"
                try:
                    sf.write(debug_audio_path, audio_chunk, self.sample_rate)
                    logger.debug(f"Saved debug audio to {debug_audio_path}")
                except Exception as e:
                    logger.error(f"Failed to save debug audio: {e}")
                
                # Detect speech
                is_speech = self._detect_speech(audio_chunk)
                logger.debug(f"VAD result (main loop): {is_speech}")
                logger.debug(f"Speech buffer length: {len(self.speech_buffer)}")
                
                if is_speech:
                    self.silence_frames = 0
                    self.speech_buffer.append(audio_chunk)
                    self.is_speaking = True
                    self.force_transcription_counter = 0  # Reset counter if speech detected
                    
                    # Check if buffer is too long - process immediately
                    if len(self.speech_buffer) >= self.max_speech_buffer_chunks:
                        logger.warning(f"Max speech buffer length reached ({self.max_speech_buffer_chunks} chunks). Processing buffer.")
                        combined_audio = np.concatenate(self.speech_buffer)
                        logger.info("Processing speech buffer due to max length.")
                        
                        # Process immediately instead of creating a separate task
                        await self._process_speech_buffer(combined_audio, "max length")
                        
                        self.speech_buffer = []
                        self.is_speaking = False
                        self.force_transcription_counter = 0
                else:
                    self.silence_frames += 1
                    self.force_transcription_counter += 1
                    
                    # Process speech buffer if silence detected - more aggressive
                    if (self.is_speaking and 
                        self.silence_frames >= self.silence_threshold and 
                        self.speech_buffer):
                        logger.info(f"Silence detected for {self.silence_frames} frames. Processing speech buffer.")
                        # Combine speech buffer
                        combined_audio = np.concatenate(self.speech_buffer)
                        logger.info("Processing speech buffer due to silence.")
                        
                        # Process immediately instead of creating a separate task
                        await self._process_speech_buffer(combined_audio, "silence")
                        
                        # Reset buffers
                        self.speech_buffer = []
                        self.is_speaking = False
                        self.force_transcription_counter = 0
                    
                    # Force transcription after N chunks if nothing detected - more aggressive
                    elif self.force_transcription_counter >= self.force_transcription_limit:
                        logger.warning(f"Force transcription triggered after {self.force_transcription_counter} chunks of silence.")
                        if self.speech_buffer:
                            combined_audio = np.concatenate(self.speech_buffer)
                            logger.info("Processing speech buffer due to forced transcription.")
                            
                            # Process immediately instead of creating a separate task
                            await self._process_speech_buffer(combined_audio, "forced")
                            
                            self.speech_buffer = []
                        else:
                            logger.warning("Force transcription attempted but speech buffer is empty.")
                        self.is_speaking = False
                        self.force_transcription_counter = 0
                        
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Audio processing error: {e}")
                import traceback
                logger.error(f"Traceback: {traceback.format_exc()}")
    
    def _audio_output_callback(self, outdata, frames, time, status):
        """Callback for audio output"""
        if status:
            logger.warning(f"Audio output status: {status}")
        try:
            # Serve audio from persistent output buffer
            if self.output_audio_buffer.size >= frames:
                chunk = self.output_audio_buffer[:frames]
                self.output_audio_buffer = self.output_audio_buffer[frames:]
                outdata[:] = chunk.reshape(-1, 1)
                logger.debug(f"Audio playback started. Served {frames} samples, {len(self.output_audio_buffer)} left in buffer. Audio range: min={chunk.min():.4f}, max={chunk.max():.4f}")
            elif self.output_audio_buffer.size > 0:
                # Serve what we have and pad with silence
                chunk = self.output_audio_buffer
                padding = np.zeros(frames - len(chunk), dtype=np.float32)
                outdata[:] = np.concatenate([chunk, padding]).reshape(-1, 1)
                self.output_audio_buffer = np.array([], dtype=np.float32)
                logger.debug(f"Audio playback (partial). Served {len(chunk)} samples, padded {len(padding)}. Audio range: min={chunk.min():.4f}, max={chunk.max():.4f}")
            else:
                # Output silence if no audio available
                outdata.fill(0)
                logger.debug("Audio output: silence (output buffer empty)")
        except Exception as e:
            logger.error(f"Audio output callback error: {e}")
        finally:
            logger.debug("Audio playback callback finished.")

    async def start(self):
        """Start the voice changer"""
        logger.info("Starting real-time voice changer...")
        self.is_running = True
        # Start audio processor in background
        processor_task = asyncio.create_task(self._audio_processor())
        try:
            # Start audio streams
            with sd.InputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype=np.float32,
                blocksize=self.chunk_size,
                callback=self._audio_callback
            ), sd.OutputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype=np.float32,
                blocksize=self.chunk_size,
                callback=self._audio_output_callback
            ):
                logger.info("Voice changer is running. Press Ctrl+C to stop.")
                # Keep running
                while self.is_running:
                    await asyncio.sleep(0.1)
        except KeyboardInterrupt:
            logger.info("Stopping voice changer...")
        except Exception as e:
            logger.error(f"Error in audio stream: {e}")
        finally:
            self.is_running = False
            processor_task.cancel()
            try:
                await processor_task
            except asyncio.CancelledError:
                pass
            except Exception as e:
                logger.error(f"Error during processor shutdown: {e}")
            logger.info("Voice changer stopped.")

# ---
# NOTE: For true streaming architectures, you would need:
# - A streaming STT API (not available in OpenAI Whisper API as of now)
# - A streaming TTS API (ElevenLabs supports streaming, but you must play audio as it arrives)
# - Overlap input/output buffers for low-latency, continuous audio
# This implementation is phrase-based, not true streaming.
# ---

async def main():
    """Main function"""
    try:
        voice_changer = RealTimeVoiceChanger()
        await voice_changer.start()
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Program terminated by user")
    except Exception as e:
        logger.error(f"Program error: {e}")
        import traceback
        traceback.print_exc() 