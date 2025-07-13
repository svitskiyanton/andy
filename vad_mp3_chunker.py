import os
import argparse
import webrtcvad
import numpy as np
from pydub import AudioSegment

# Settings
VAD_SAMPLE_RATE = 16000
VAD_FRAME_DURATION = 0.03  # 30ms
VAD_FRAME_SIZE = int(VAD_SAMPLE_RATE * VAD_FRAME_DURATION)
MIN_PHRASE_DURATION = 0.2  # seconds
SILENCE_DURATION = 0.05    # seconds (50ms)


def detect_vad_boundaries(audio_segment):
    """Detect phrase boundaries using VAD and return list of (start_ms, end_ms) tuples."""
    vad = webrtcvad.Vad(0)  # Most sensitive
    audio_16k = audio_segment.set_frame_rate(VAD_SAMPLE_RATE).set_channels(1)
    samples = np.array(audio_16k.get_array_of_samples(), dtype=np.int16)
    frame_size = int(VAD_SAMPLE_RATE * VAD_FRAME_DURATION)
    total_frames = len(samples) // frame_size

    speech_flags = []
    for i in range(total_frames):
        frame = samples[i*frame_size:(i+1)*frame_size]
        if len(frame) < frame_size:
            break
        is_speech = vad.is_speech(frame.tobytes(), VAD_SAMPLE_RATE)
        speech_flags.append(is_speech)

    # Find boundaries: split at runs of silence
    boundaries = []
    in_speech = False
    chunk_start = 0
    for i, is_speech in enumerate(speech_flags):
        t = i * VAD_FRAME_DURATION
        if is_speech and not in_speech:
            # Start of speech
            chunk_start = t
            in_speech = True
        elif not is_speech and in_speech:
            # End of speech (pause)
            chunk_end = t
            if chunk_end - chunk_start >= MIN_PHRASE_DURATION:
                boundaries.append((int(chunk_start*1000), int(chunk_end*1000)))
            in_speech = False
    # Handle last chunk
    if in_speech:
        chunk_end = total_frames * VAD_FRAME_DURATION
        if chunk_end - chunk_start >= MIN_PHRASE_DURATION:
            boundaries.append((int(chunk_start*1000), int(chunk_end*1000)))
    return boundaries


def main():
    parser = argparse.ArgumentParser(description="Split MP3 into chunks at natural pauses using VAD.")
    parser.add_argument("input_file", help="Input MP3 file")
    parser.add_argument("--outdir", default="chunks_vad", help="Output directory for chunks")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    audio = AudioSegment.from_mp3(args.input_file)
    print(f"Loaded {args.input_file}: {len(audio)/1000:.2f}s")

    boundaries = detect_vad_boundaries(audio)
    print(f"Detected {len(boundaries)} chunks:")
    for i, (start, end) in enumerate(boundaries, 1):
        print(f"  Chunk {i}: {start/1000:.2f}s - {end/1000:.2f}s ({(end-start)/1000:.2f}s)")
        chunk = audio[start:end]
        out_path = os.path.join(args.outdir, f"chunk_{i}.mp3")
        chunk.export(out_path, format="mp3")
    print(f"Saved {len(boundaries)} chunks to {args.outdir}/")

if __name__ == "__main__":
    main() 