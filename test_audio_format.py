import os
import numpy as np
import soundfile as sf
from elevenlabs import generate, set_api_key
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Setup API key
api_key = os.getenv('ELEVENLABS_API_KEY')
if not api_key:
    print("❌ ELEVENLABS_API_KEY not found in environment variables")
    exit(1)
set_api_key(api_key)

def test_elevenlabs_audio():
    """Test what format ElevenLabs generates"""
    print("Testing ElevenLabs audio generation...")
    
    try:
        # Generate a simple test audio
        audio_stream = generate(
            text="Test audio",
            voice="21m00Tcm4TlvDq8ikWAM",
            model="eleven_flash_v2_5",
            stream=True
        )
        
        # Collect chunks
        audio_chunks = []
        for chunk in audio_stream:
            audio_chunks.append(chunk)
        
        if audio_chunks:
            # Combine chunks
            combined_audio = b''.join(audio_chunks)
            print(f"Raw audio length: {len(combined_audio)} bytes")
            
            # Try different interpretations
            if len(combined_audio) % 2 == 0:
                # Try as int16
                audio_int16 = np.frombuffer(combined_audio, dtype=np.int16)
                print(f"As int16: {len(audio_int16)} samples")
                print(f"Audio range: min={audio_int16.min()}, max={audio_int16.max()}")
                
                # Convert to float
                audio_float = audio_int16.astype(np.float32) / 32767.0
                print(f"As float: min={audio_float.min():.4f}, max={audio_float.max():.4f}")
                
                # Save for inspection
                sf.write("test_elevenlabs_original.wav", audio_float, 44100)
                print("Saved as test_elevenlabs_original.wav at 44.1kHz")
                
                # Try different sample rates
                for target_sr in [16000, 22050, 44100]:
                    target_length = int(len(audio_float) * target_sr / 44100)
                    resampled = np.interp(
                        np.linspace(0, len(audio_float) - 1, target_length),
                        np.arange(len(audio_float)),
                        audio_float
                    )
                    sf.write(f"test_elevenlabs_{target_sr}hz.wav", resampled, target_sr)
                    print(f"Saved as test_elevenlabs_{target_sr}hz.wav")
                
            else:
                print("Audio length is not even, cannot interpret as int16")
                
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_elevenlabs_audio() 