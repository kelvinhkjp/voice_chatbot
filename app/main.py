import os
import shutil
import tempfile
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from dotenv import load_dotenv

# Load env variables from .env file
# Use explicit path so the app works regardless of CWD
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(_project_root, ".env"))

# Initialize Chatbot Engine
from app.chat_engine import ChatbotEngine
try:
    chatbot = ChatbotEngine()
except Exception as e:
    print(f"Warning: ChatbotEngine initialization failed (most likely due to missing GEMINI_API_KEY): {e}")
    chatbot = None

app = FastAPI(title="Voice Chatbot Server")

# Enable CORS for local development and testing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Route to serve the main HTML file at root
@app.get("/")
async def get_index():
    index_path = os.path.join("static", "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return HTMLResponse("<h2>Frontend static files not found. Please create static/index.html</h2>")

@app.post("/api/chat-voice")
async def chat_voice(
    audio: UploadFile = File(...),
    session_id: str = Form("default_session")
):
    """
    Upload an audio file, transcribe it, run diarization, and get a response from Gemini.
    """
    # 1. Check API Keys
    asr_engine_type = os.getenv("ASR_ENGINE", "local").lower()
    openai_key = os.getenv("OPENAI_API_KEY")
    gemini_key = os.getenv("GEMINI_API_KEY")
    hf_token = os.getenv("HF_TOKEN")
    
    if not gemini_key:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Missing GEMINI_API_KEY in .env file."
        )
        
    if asr_engine_type == "api" and not openai_key:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Missing OPENAI_API_KEY in .env file when ASR_ENGINE is set to 'api'."
        )

    # 2. Save uploaded audio to a temporary file
    temp_dir = tempfile.gettempdir()
    # We preserve the original extension (e.g. .wav, .webm)
    ext = os.path.splitext(audio.filename)[1] if audio.filename else ".wav"
    if not ext:
        ext = ".wav"
    temp_file_path = os.path.join(temp_dir, f"voice_{session_id}{ext}")
    
    try:
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(audio.file, buffer)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to save temporary audio file: {e}"
        )

    # 3. Process voice (Transcription + Diarization)
    try:
        from app.voice_pipeline import process_voice_audio
        # Run ASR and Diarization. Pass Hugging Face token if present
        aligned_dialogue = process_voice_audio(temp_file_path, hf_token or "")
    except Exception as e:
        # Clean up temp file before raising error
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Speech processing error: {e}"
        )
    finally:
        # Clean up temp file
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

    # 4. Get response from LLM
    if not chatbot:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Chatbot engine is not initialized. Check server logs."
        )
        
    try:
        response_text = await chatbot.get_response(session_id, aligned_dialogue)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"LLM processing error: {e}"
        )

    return JSONResponse({
        "aligned_dialogue": aligned_dialogue,
        "response": response_text
    })

@app.post("/api/clear-session")
async def clear_session(session_id: str = Form(...)):
    """
    Clears conversation history for the given session.
    """
    if chatbot:
        chatbot.clear_session(session_id)
        return {"status": "success", "message": f"Session {session_id} history cleared."}
    else:
        raise HTTPException(status_code=500, detail="Chatbot engine not initialized.")

# Mount the static directory to serve images, JS, and CSS
if not os.path.exists("static"):
    os.makedirs("static")
app.mount("/static", StaticFiles(directory="static"), name="static")
