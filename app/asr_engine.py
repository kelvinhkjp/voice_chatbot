import os
from abc import ABC, abstractmethod
from typing import List, Dict, Any

class ASREngine(ABC):
    """
    Abstract Base Class defining the contract for Speech-to-Text (ASR) engines.
    Any new ASR engine (e.g. Deepgram, AssemblyAI, Google ASR) should inherit
    from this and implement the transcribe method.
    """
    @abstractmethod
    def transcribe(self, audio_file_path: str) -> List[Dict[str, Any]]:
        """
        Transcribes the given audio file and returns a list of segments with timestamps.
        
        Format:
        [
            {"start": float, "end": float, "text": str},
            ...
        ]
        """
        pass


class WhisperAPIASREngine(ASREngine):
    """
    ASR Engine using the official OpenAI Whisper API.
    """
    def __init__(self):
        from openai import OpenAI
        self.client = OpenAI()

    def transcribe(self, audio_file_path: str) -> List[Dict[str, Any]]:
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY is not set in environment variables.")

        with open(audio_file_path, "rb") as audio_file:
            response = self.client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                response_format="verbose_json",
                timestamp_granularities=["segment"]
            )
            
        segments = getattr(response, "segments", [])
        
        formatted_segments = []
        for seg in segments:
            formatted_segments.append({
                "start": seg.get("start", 0.0),
                "end": seg.get("end", 0.0),
                "text": seg.get("text", "").strip()
            })
        return formatted_segments


class LocalWhisperASREngine(ASREngine):
    """
    ASR Engine running a highly optimized Whisper model locally using faster-whisper.
    """
    def __init__(self, model_size: str = "base"):
        from faster_whisper import WhisperModel
        
        # We can detect if a GPU is available. If not, default to CPU.
        # "int8" quantization is standard and fast for CPU processing.
        device = "cpu"
        compute_type = "int8"
        
        print(f"Initializing local Whisper model '{model_size}' on device: {device}...")
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)

    def transcribe(self, audio_file_path: str) -> List[Dict[str, Any]]:
        # Transcribe audio file
        # beam_size=5 is standard for a good speed/accuracy trade-off
        segments, info = self.model.transcribe(audio_file_path, beam_size=5)
        
        formatted_segments = []
        for seg in segments:
            formatted_segments.append({
                "start": seg.start,
                "end": seg.end,
                "text": seg.text.strip()
            })
        return formatted_segments


def get_asr_engine(engine_type: str = None) -> ASREngine:
    """
    Factory function to retrieve the configured ASR engine.
    Allows dynamic choice without changing client code.
    """
    if engine_type is None:
        # Load configuration from environment variable, default to local
        engine_type = os.getenv("ASR_ENGINE", "local").lower()
        
    if engine_type == "local":
        model_size = os.getenv("LOCAL_WHISPER_MODEL", "base")
        return LocalWhisperASREngine(model_size=model_size)
    elif engine_type == "api":
        return WhisperAPIASREngine()
    else:
        raise ValueError(f"Unknown ASR engine type: {engine_type}")
