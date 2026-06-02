import os
import subprocess
from typing import List, Dict, Any
from app.asr_engine import get_asr_engine

def convert_to_wav(audio_file_path: str) -> str:
    """
    Converts audio file to .wav format using FFmpeg if needed.
    Returns the path to the .wav file.
    """
    if audio_file_path.endswith(".wav"):
        return audio_file_path  # Already wav, no conversion needed
    
    wav_path = audio_file_path.rsplit(".", 1)[0] + ".wav"
    subprocess.run([
        "ffmpeg", "-y", "-i", audio_file_path,
        "-ar", "16000", "-ac", "1", wav_path
    ], check=True, capture_output=True)
    print(f"Converted {audio_file_path} → {wav_path}")
    return wav_path

def run_diarization(audio_file_path: str, hf_token: str) -> List[Dict[str, Any]]:
    """
    Runs pyannote-audio speaker diarization locally.
    Returns a list of speaker turns: [{"start": float, "end": float, "speaker": str}]
    """
    # Import locally to avoid slow imports if pyannote is not initialized
    import torch
    import soundfile as sf
    from pyannote.audio import Pipeline
    
    try:
        # Load the pretrained diarization pipeline
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            token=hf_token
        )

        # Pre-load audio with soundfile to bypass torchcodec/FFmpeg DLL issue
        data, sample_rate = sf.read(audio_file_path, dtype="float32", always_2d=True)
        waveform = torch.tensor(data.T)  # shape: [channels, samples]
        audio_input = {"waveform": waveform, "sample_rate": int(sample_rate)}
        
        print(f"Waveform shape: {waveform.shape}, Sample rate: {sample_rate}")

        # Run diarization on the file
        diarization = pipeline(audio_input)
        
        speaker_turns = []

        annotation = diarization.speaker_diarization
        for turn, _, speaker in annotation.itertracks(yield_label=True):
            speaker_turns.append({
                "start": turn.start,
                "end": turn.end,
                "speaker": speaker
            })

        return speaker_turns
    except Exception as e:
        print(f"Error during speaker diarization: {e}")
        # Fallback to single speaker if PyAnnote fails (e.g. HF_TOKEN is invalid)
        return []

def align_transcript_and_speakers(
    whisper_segments: List[Dict[str, Any]], 
    diarization_turns: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Aligns Whisper transcript segments with pyannote speaker intervals by calculating
    the maximum time overlap for each text segment.
    """
    aligned_dialogue = []
    
    # If diarization failed or returned no speaker segments, default everyone to "Speaker"
    if not diarization_turns:
        for seg in whisper_segments:
            aligned_dialogue.append({
                "speaker": "Speaker",
                "text": seg["text"],
                "start": seg["start"],
                "end": seg["end"]
            })
        return aligned_dialogue

    for w_seg in whisper_segments:
        w_start = w_seg["start"]
        w_end = w_seg["end"]
        
        best_speaker = "Speaker Unknown"
        max_overlap = 0.0
        
        # Check overlaps with all diarization intervals
        for d_turn in diarization_turns:
            d_start = d_turn["start"]
            d_end = d_turn["end"]
            
            # Calculate overlapping duration
            overlap_start = max(w_start, d_start)
            overlap_end = min(w_end, d_end)
            overlap_duration = max(0.0, overlap_end - overlap_start)
            
            if overlap_duration > max_overlap:
                max_overlap = overlap_duration
                best_speaker = d_turn["speaker"]
                
        # If overlap is extremely small or zero, fallback to the speaker active at the start
        if max_overlap == 0.0:
            # Find the closest speaker interval
            closest_turn = min(
                diarization_turns,
                key=lambda x: min(abs(x["start"] - w_start), abs(x["end"] - w_start))
            )
            best_speaker = closest_turn["speaker"]
            
        aligned_dialogue.append({
            "speaker": best_speaker,
            "text": w_seg["text"],
            "start": w_start,
            "end": w_end
        })
        
    return aligned_dialogue

def process_voice_audio(audio_file_path: str, hf_token: str) -> List[Dict[str, Any]]:
    """
    Coordinates transcription, diarization, and alignment.
    """
    # 0. Convert to wav first so both Whisper and pyannote can read it
    audio_file_path = convert_to_wav(audio_file_path)

    # 1. Transcribe dynamically using the configured ASR Engine
    asr_engine = get_asr_engine()
    whisper_segs = asr_engine.transcribe(audio_file_path)
    print(f"DEBUG Whisper segments: {whisper_segs}")
    # 2. Diarize (via local pyannote-audio)
    diarization_turns = run_diarization(audio_file_path, hf_token)
    print(f"DEBUG Diarization turns: {diarization_turns}") 
    # 3. Align
    aligned_dialogue = align_transcript_and_speakers(whisper_segs, diarization_turns)
    print(f"DEBUG Aligned dialogue: {aligned_dialogue}")
    return aligned_dialogue
