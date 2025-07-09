#!/usr/bin/env python3
"""
Real-time Voice Changer for Windows
Uses ElevenLabs Pro with eleven_flash_v2_5 model and Google credentials
Optimized for Windows with available dependencies
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
import webrtcvad_wheels as webrtcvad

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RealTimeVoiceChanger:
    def __init__(self):
        # Audio settings
        self.sample_rate = 16000
        self.chunk_duration = 0.5  # seconds
        self.chunk_size = int(self.sample_rate * self.chunk_duration)
        self.channels = 1
        
        # Voice settings
        self.target_voice_id = "21m00Tcm4TlvDq8ikWAM"  # Rachel (ElevenLabs)
        self.elevenlabs_model = "eleven_flash_v2_5"
        
        # VAD settings
        self.vad = webrtcvad.Vad(2)  # Aggressiveness level 2
        self.frame_duration = 30  # ms
        
        # Queues for audio processing
        self.input_queue = queue.Queue(maxsize=10)
        self.output_queue = queue.Queue(maxsize=10)
        self.text_queue = queue.Queue(maxsize=5)
        
        # State management
        self.is_running = False
        self.is_speaking = False
        self.speech_buffer = []
        self.silence_frames = 0
        self.silence_threshold = 10  # frames of silence before processing
        
        # Initialize APIs
        self._setup_apis()
        
    def _setup_apis(self):
        """Setup API keys and clients"""
        # ElevenLabs
        elevenlabs_key = os.getenv('ELEVENLABS_API_KEY')
        if not elevenlabs_key:
            raise ValueError("ELEVENLABS_API_KEY not found in environment variables")
        set_api_key(elevenlabs_key)
        
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
        
        # Add to input queue
        try:
            self.input_queue.put_nowait(audio_data)
        except queue.Full:
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
            # Convert to bytes for VAD
            audio_bytes = (audio_chunk * 32767).astype(np.int16).tobytes()
            
            # Check each frame
            frame_size = int(self.sample_rate * self.frame_duration / 1000)
            speech_detected = False
            
            for i in range(0, len(audio_bytes) - frame_size, frame_size):
                frame = audio_chunk[i:i + frame_size]
                if len(frame) == frame_size:
                    is_speech = self.vad.is_speech(frame, self.sample_rate)
                    if is_speech:
                        speech_detected = True
                        break
            
            return speech_detected
        except Exception as e:
            logger.error(f"VAD error: {e}")
            return True  # Default to speech if VAD fails
    
    async def _transcribe_audio(self, audio_chunk):
        """Transcribe audio using OpenAI Whisper"""
        try:
            # Save audio to temporary file
            temp_file = "temp_audio.wav"
            sf.write(temp_file, audio_chunk, self.sample_rate)
            
            # Transcribe with OpenAI
            with open(temp_file, "rb") as audio_file:
                response = openai.Audio.transcribe(
                    model="whisper-1",
                    file=audio_file,
                    language="en"
                )
            
            # Clean up
            os.remove(temp_file)
            
            text = response.get("text", "").strip()
            if text:
                logger.info(f"Transcribed: {text}")
                return text
            
        except Exception as e:
            logger.error(f"Transcription error: {e}")
        
        return None
    
    async def _generate_speech(self, text):
        """Generate speech using ElevenLabs"""
        try:
            # Generate audio with ElevenLabs
            audio_stream = generate(
                text=text,
                voice=self.target_voice_id,
                model=self.elevenlabs_model,
                stream=True
            )
            
            # Convert stream to audio data
            audio_chunks = []
            for chunk in audio_stream:
                audio_chunks.append(chunk)
            
            if audio_chunks:
                # Combine chunks and convert to numpy array
                combined_audio = b''.join(audio_chunks)
                audio_array = np.frombuffer(combined_audio, dtype=np.int16)
                audio_float = audio_array.astype(np.float32) / 32767.0
                
                logger.info(f"Generated speech for: {text[:50]}...")
                return audio_float
            
        except Exception as e:
            logger.error(f"Speech generation error: {e}")
        
        return None
    
    async def _audio_processor(self):
        """Main audio processing loop"""
        logger.info("Starting audio processor...")
        
        while self.is_running:
            try:
                # Get audio chunk from input queue
                audio_chunk = self.input_queue.get(timeout=1.0)
                
                # Detect speech
                is_speech = self._detect_speech(audio_chunk)
                
                if is_speech:
                    self.silence_frames = 0
                    self.speech_buffer.append(audio_chunk)
                    self.is_speaking = True
                else:
                    self.silence_frames += 1
                    
                    # Process speech buffer if silence detected
                    if (self.is_speaking and 
                        self.silence_frames >= self.silence_threshold and 
                        self.speech_buffer):
                        
                        # Combine speech buffer
                        combined_audio = np.concatenate(self.speech_buffer)
                        
                        # Transcribe
                        text = await self._transcribe_audio(combined_audio)
                        
                        if text:
                            # Generate speech
                            output_audio = await self._generate_speech(text)
                            
                            if output_audio is not None:
                                # Add to output queue
                                try:
                                    self.output_queue.put_nowait(output_audio)
                                except queue.Full:
                                    logger.warning("Output queue full, dropping audio")
                        
                        # Reset buffers
                        self.speech_buffer = []
                        self.is_speaking = False
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Audio processing error: {e}")
    
    def _audio_output_callback(self, outdata, frames, time, status):
        """Callback for audio output"""
        if status:
            logger.warning(f"Audio output status: {status}")
        
        try:
            # Get audio from output queue
            audio_chunk = self.output_queue.get_nowait()
            
            # Ensure correct length
            if len(audio_chunk) < frames:
                # Pad with silence
                padding = np.zeros(frames - len(audio_chunk))
                audio_chunk = np.concatenate([audio_chunk, padding])
            elif len(audio_chunk) > frames:
                # Truncate
                audio_chunk = audio_chunk[:frames]
            
            # Convert to stereo if needed
            if outdata.shape[1] > 1:
                outdata[:] = audio_chunk.reshape(-1, 1)
            else:
                outdata[:] = audio_chunk.reshape(-1, 1)
                
        except queue.Empty:
            # Output silence if no audio available
            outdata.fill(0)
    
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
            
            logger.info("Voice changer stopped.")

async def main():
    """Main function"""
    try:
        voice_changer = RealTimeVoiceChanger()
        await voice_changer.start()
    except Exception as e:
        logger.error(f"Fatal error: {e}")

if __name__ == "__main__":
    asyncio.run(main()) 