import os
import base64
import google.generativeai as genai
from gtts import gTTS
from PIL import Image
import wave
import struct
import io
import time
import whisper
from pydub import AudioSegment
import tempfile
import subprocess
import sys
import librosa
import soundfile as sf
import numpy as np
from dotenv import load_dotenv
import platform
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import shutil
import uuid
from pathlib import Path

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="AI4LI Image & Audio Processing API",
    description="API for processing images with Gemini AI and transcribing audio with Whisper",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
OUTPUT_DIR = "api_output"
UPLOAD_DIR = "uploads"

# Create directories
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Platform-specific FFmpeg configuration
def get_ffmpeg_path():
    """Get FFmpeg path based on the operating system."""
    system = platform.system().lower()
    
    if system == "darwin":  # macOS
        # Try common Homebrew locations
        homebrew_paths = [
            "/opt/homebrew/bin",  # Apple Silicon Macs
            "/usr/local/bin"      # Intel Macs
        ]
        
        for path in homebrew_paths:
            if os.path.exists(os.path.join(path, "ffmpeg")):
                return path
        
        # If not found in Homebrew locations, assume it's in PATH
        return None
        
    elif system == "windows":
        # Windows FFmpeg path (original)
        return r"E:\ffmpeg-master-latest-win64-gpl-shared\ffmpeg-master-latest-win64-gpl-shared\bin"
    
    else:  # Linux and others
        return None  # Assume ffmpeg is in PATH

def get_ffmpeg_executable(executable_name):
    """Get the full path to FFmpeg executable based on OS."""
    system = platform.system().lower()
    
    if system == "windows":
        return f"{executable_name}.exe"
    else:
        return executable_name

# Set FFmpeg path
FFMPEG_PATH = get_ffmpeg_path()
if FFMPEG_PATH and FFMPEG_PATH not in os.environ["PATH"]:
    os.environ["PATH"] = FFMPEG_PATH + os.pathsep + os.environ["PATH"]

# Configure Gemini AI
genai.configure(api_key=GOOGLE_API_KEY)

# Initialize Whisper model
try:
    print("Loading Whisper model...")
    whisper_model = whisper.load_model("base")
    print("Whisper model loaded successfully")
except Exception as e:
    print(f"Error initializing Whisper model: {e}")
    whisper_model = None

# Pydantic models
class ProcessImageRequest(BaseModel):
    information_needs: str = "tên, màu sắc, loại sản phẩm và hạn sử dụng"

class ProcessResponse(BaseModel):
    success: bool
    message: str
    description: Optional[str] = None
    transcription: Optional[str] = None
    audio_file: Optional[str] = None
    text_file: Optional[str] = None

# Helper functions
def encode_image(image_path):
    """Encode image to base64 string."""
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode("utf-8")
    except Exception as e:
        print(f"Error encoding image: {e}")
        return None

def process_with_gemini(image_path, information_needs):
    """Process image with Gemini API and return description."""
    if not os.path.exists(image_path):
        raise HTTPException(status_code=404, detail=f"Image file not found at {image_path}")
        
    base64_image = encode_image(image_path)
    if not base64_image:
        raise HTTPException(status_code=500, detail="Could not encode image")

    try:
        model = genai.GenerativeModel("gemini-1.5-flash-latest")
        prompt = (
           f"Nhận diện sản phẩm trong ảnh, hãy quan sát thông tin của sản phẩm và đưa ra thông tin về {information_needs}. Lưu ý: trả lời bằng tiếng Việt và giữ câu trả lời đơn giản, ngắn gọn và cấu trúc của một câu phải có chủ ngữ và vị ngữ."
        )
        response = model.generate_content([
            {"mime_type": "image/jpeg", "data": base64_image},
            prompt
        ], generation_config={"max_output_tokens": 100})
        
        return response.text
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing with Gemini: {e}")

def check_ffmpeg():
    """Check if FFmpeg is installed and accessible."""
    try:
        result = subprocess.run(['ffmpeg', '-version'], 
                              capture_output=True, 
                              text=True,
                              check=True)
        return True
    except (subprocess.SubprocessError, FileNotFoundError):
        return False

def convert_audio_to_wav(input_path, output_path):
    """Convert audio file to WAV format using pydub."""
    try:
        if not check_ffmpeg():
            raise HTTPException(status_code=500, detail="FFmpeg not found or not accessible")

        try:
            # Set FFmpeg paths based on OS
            if FFMPEG_PATH:
                AudioSegment.converter = os.path.join(FFMPEG_PATH, get_ffmpeg_executable("ffmpeg"))
                AudioSegment.ffmpeg = os.path.join(FFMPEG_PATH, get_ffmpeg_executable("ffmpeg"))
                AudioSegment.ffprobe = os.path.join(FFMPEG_PATH, get_ffmpeg_executable("ffprobe"))
            
            audio = AudioSegment.from_file(input_path)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error loading audio file: {str(e)}")
        
        audio = audio.set_frame_rate(16000).set_channels(1)

        try:
            audio.export(output_path, format="wav")
            return True
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error saving WAV file: {str(e)}")
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error converting audio: {str(e)}")

def transcribe_audio(audio_path):
    """Transcribe audio file using Whisper model."""
    if not whisper_model:
        raise HTTPException(status_code=500, detail="Whisper model not available")
        
    if not os.path.exists(audio_path):
        raise HTTPException(status_code=404, detail=f"Audio file not found at {audio_path}")
            
    try:
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_wav:
            temp_wav_path = temp_wav.name
            convert_audio_to_wav(audio_path, temp_wav_path)
            
            if not os.path.exists(temp_wav_path) or os.path.getsize(temp_wav_path) == 0:
                raise HTTPException(status_code=500, detail="WAV file was not created properly")
                
            result = whisper_model.transcribe(temp_wav_path, language="vi")
            transcription = result["text"]
            
            try:
                os.unlink(temp_wav_path)
            except Exception:
                pass
            
            return transcription
                
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during transcription: {e}")

def save_description_to_file(description, output_path):
    """Save the description to a text file."""
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(description)
        return True
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error saving description to file: {e}")

def clean_text_for_audio(text):
    """Clean text by removing markdown and special characters for better audio generation."""
    text = text.replace('*', '').replace('_', '').replace('**', '')
    text = ' '.join(text.split())
    return text

def generate_audio_response(text, output_path):
    """Generate audio response from text using gTTS."""
    try:
        cleaned_text = clean_text_for_audio(text)
        tts = gTTS(text=cleaned_text, lang="vi")
        tts.save(output_path)
        return True
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating audio response: {e}")

# API Endpoints
@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Naviblind API",
        "version": "1.0.0",
        "platform": f"{platform.system()} {platform.release()}",
        "ffmpeg_path": FFMPEG_PATH or "Using system PATH",
        "whisper_available": whisper_model is not None
    }

@app.post("/process-image", response_model=ProcessResponse)
async def process_image_endpoint(
    image: UploadFile = File(...),
    information_needs: str = Form("tên, màu sắc, loại sản phẩm và hạn sử dụng")
):
    """Process an uploaded image with Gemini AI."""
    if not image.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    # Generate unique filename
    file_id = str(uuid.uuid4())
    image_path = os.path.join(UPLOAD_DIR, f"{file_id}_{image.filename}")
    
    try:
        # Save uploaded image
        with open(image_path, "wb") as buffer:
            shutil.copyfileobj(image.file, buffer)
        
        # Process with Gemini
        description = process_with_gemini(image_path, information_needs)
        
        # Save description to file
        text_file = os.path.join(OUTPUT_DIR, f"{file_id}_description.txt")
        save_description_to_file(description, text_file)
        
        # Generate audio response
        audio_file = os.path.join(OUTPUT_DIR, f"{file_id}_response.mp3")
        generate_audio_response(description, audio_file)
        
        return ProcessResponse(
            success=True,
            message="Image processed successfully",
            description=description,
            audio_file=f"/download/audio/{file_id}_response.mp3",
            text_file=f"/download/text/{file_id}_description.txt"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Clean up uploaded image
        if os.path.exists(image_path):
            os.remove(image_path)

@app.post("/transcribe-audio", response_model=ProcessResponse)
async def transcribe_audio_endpoint(audio: UploadFile = File(...)):
    """Transcribe an uploaded audio file."""
    if not audio.content_type.startswith('audio/'):
        raise HTTPException(status_code=400, detail="File must be an audio file")
    
    # Generate unique filename
    file_id = str(uuid.uuid4())
    audio_path = os.path.join(UPLOAD_DIR, f"{file_id}_{audio.filename}")
    
    try:
        # Save uploaded audio
        with open(audio_path, "wb") as buffer:
            shutil.copyfileobj(audio.file, buffer)
        
        # Transcribe audio
        transcription = transcribe_audio(audio_path)
        
        # Save transcription to file
        text_file = os.path.join(OUTPUT_DIR, f"{file_id}_transcription.txt")
        save_description_to_file(transcription, text_file)
        
        return ProcessResponse(
            success=True,
            message="Audio transcribed successfully",
            transcription=transcription,
            text_file=f"/download/text/{file_id}_transcription.txt"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Clean up uploaded audio
        if os.path.exists(audio_path):
            os.remove(audio_path)

@app.post("/process-combined", response_model=ProcessResponse)
async def process_combined_endpoint(
    image: UploadFile = File(...),
    audio: UploadFile = File(...),
):
    """Process both image and audio files together."""
    if not image.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="Image file must be an image")
    
    if not audio.content_type.startswith('audio/'):
        raise HTTPException(status_code=400, detail="Audio file must be an audio file")
    
    # Generate unique filename
    file_id = str(uuid.uuid4())
    image_path = os.path.join(UPLOAD_DIR, f"{file_id}_{image.filename}")
    audio_path = os.path.join(UPLOAD_DIR, f"{file_id}_{audio.filename}")
    
    try:
        # Save uploaded files
        with open(image_path, "wb") as buffer:
            shutil.copyfileobj(image.file, buffer)
        
        with open(audio_path, "wb") as buffer:
            shutil.copyfileobj(audio.file, buffer)
        
        # Transcribe audio to get information needs
        transcription = transcribe_audio(audio_path)
        information_needs = transcription if transcription else "tên, màu sắc, loại sản phẩm và hạn sử dụng"
        
        # Process image with Gemini
        description = process_with_gemini(image_path, information_needs)
        
        # Save files
        text_file = os.path.join(OUTPUT_DIR, f"{file_id}_result.txt")
        combined_text = f"Transcription: {transcription}\n\nDescription: {description}"
        save_description_to_file(combined_text, text_file)
        
        # Generate audio response
        audio_file = os.path.join(OUTPUT_DIR, f"{file_id}_response.mp3")
        generate_audio_response(description, audio_file)
        
        return ProcessResponse(
            success=True,
            message="Combined processing completed successfully",
            description=description,
            transcription=transcription,
            audio_file=f"/download/audio/{file_id}_response.mp3",
            text_file=f"/download/text/{file_id}_result.txt"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Clean up uploaded files
        for path in [image_path, audio_path]:
            if os.path.exists(path):
                os.remove(path)

@app.get("/download/audio/{filename}")
async def download_audio(filename: str):
    """Download generated audio file."""
    file_path = os.path.join(OUTPUT_DIR, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Audio file not found")
    return FileResponse(file_path, media_type="audio/mpeg", filename=filename)

@app.get("/download/text/{filename}")
async def download_text(filename: str):
    """Download generated text file."""
    file_path = os.path.join(OUTPUT_DIR, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Text file not found")
    return FileResponse(file_path, media_type="text/plain", filename=filename)

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "ffmpeg_available": check_ffmpeg(),
        "whisper_available": whisper_model is not None,
        "gemini_configured": GOOGLE_API_KEY is not None
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 