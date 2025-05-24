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

# Load environment variables
load_dotenv()

# Configuration
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
OUTPUT_DIR = r"C:\Users\ACER\Documents\AI4LI"
IMAGE_PATH = r"C:\Users\ACER\Documents\AI4LI\20250317_131158.jpg"
AUDIO_PATH = r"C:\Users\ACER\Documents\AI4LI\record.m4a"

# Add FFmpeg path
FFMPEG_PATH = r"E:\ffmpeg-master-latest-win64-gpl-shared\ffmpeg-master-latest-win64-gpl-shared\bin"
if FFMPEG_PATH not in os.environ["PATH"]:
    os.environ["PATH"] = FFMPEG_PATH + os.pathsep + os.environ["PATH"]

# Initialize Gemini
genai.configure(api_key=GOOGLE_API_KEY)

# Initialize Whisper model
try:
    print("Loading Whisper model...")
    whisper_model = whisper.load_model("base")
    print("Whisper model loaded successfully")
except Exception as e:
    print(f"Error initializing Whisper model: {e}")
    print("Please install the correct dependencies:")
    print("pip install openai-whisper")
    exit(1)

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
        print(f"Error: Image file not found at {image_path}")
        return None
        
    base64_image = encode_image(image_path)
    if not base64_image:
        print("Error: Could not encode image.")
        return None

    try:
        model = genai.GenerativeModel("gemini-1.5-flash-latest")
        prompt = (
           f"Nhận diện sản phẩm trong ảnh, hãy quan sát thông tin của sản phẩm và đưa ra thông tin về {information_needs}. Lưu ý: trả lời bằng tiếng Việt và trả lời chi tiết cấu trúc của một câu có chủ ngữ và vị ngữ."
        )
        response = model.generate_content([
            {"mime_type": "image/jpeg", "data": base64_image},
            prompt
        ], generation_config={"max_output_tokens": 100})
        
        return response.text
    except Exception as e:
        print(f"Error processing with Gemini: {e}")
        return None

def check_ffmpeg():
    """Check if FFmpeg is installed and accessible."""
    try:
        # Try to run ffmpeg -version
        result = subprocess.run(['ffmpeg', '-version'], 
                              capture_output=True, 
                              text=True,
                              check=True)
        print("FFmpeg is installed and accessible")
        print(f"FFmpeg version: {result.stdout.split()[2]}")
        return True
    except (subprocess.SubprocessError, FileNotFoundError) as e:
        print(f"FFmpeg error: {str(e)}")
        print("\nPlease verify FFmpeg installation:")
        print(f"1. Check if FFmpeg exists at: {FFMPEG_PATH}")
        print("2. Make sure these files exist:")
        print(f"   - {os.path.join(FFMPEG_PATH, 'ffmpeg.exe')}")
        print(f"   - {os.path.join(FFMPEG_PATH, 'ffprobe.exe')}")
        print("3. Try adding FFmpeg to PATH manually:")
        print("   - Open System Properties")
        print("   - Click 'Environment Variables'")
        print("   - Under 'System variables', find 'Path'")
        print("   - Add the FFmpeg bin folder path")
        return False

def convert_audio_to_wav(input_path, output_path):
    """Convert audio file to WAV format using pydub."""
    try:
        if not check_ffmpeg():
            return False

        print(f"Loading audio file: {input_path}")
        
        # Load M4A file
        try:
            # Set FFmpeg path for pydub
            AudioSegment.converter = os.path.join(FFMPEG_PATH, "ffmpeg.exe")
            AudioSegment.ffmpeg = os.path.join(FFMPEG_PATH, "ffmpeg.exe")
            AudioSegment.ffprobe = os.path.join(FFMPEG_PATH, "ffprobe.exe")
            
            audio = AudioSegment.from_file(input_path, format="m4a")
        except Exception as e:
            print(f"Error loading M4A file: {str(e)}")
            return False
        
        # Convert to mono and set sample rate
        print("Converting audio format...")
        audio = audio.set_frame_rate(16000).set_channels(1)
        
        # Export as WAV
        print("Saving as WAV...")
        try:
            audio.export(output_path, format="wav")
            print("Conversion completed successfully")
            return True
        except Exception as e:
            print(f"Error saving WAV file: {str(e)}")
            return False
        
    except Exception as e:
        print(f"Error converting audio: {str(e)}")
        return False

def transcribe_audio(audio_path):
    """Transcribe audio file using Whisper model."""
    try:
        if not os.path.exists(audio_path):
            print(f"Error: Audio file not found at {audio_path}")
            return None
            
        print(f"Transcribing audio from: {audio_path}")
        
        # Convert audio to WAV format
        try:
            # Create a temporary WAV file
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_wav:
                temp_wav_path = temp_wav.name
                
                # Convert audio using pydub
                if convert_audio_to_wav(audio_path, temp_wav_path):
                    # Verify the WAV file was created
                    if not os.path.exists(temp_wav_path) or os.path.getsize(temp_wav_path) == 0:
                        print("Error: WAV file was not created properly")
                        return None
                        
                    # Transcribe using Whisper
                    print("Starting transcription with Whisper...")
                    result = whisper_model.transcribe(temp_wav_path, language="vi")
                    transcription = result["text"]
                    print(f"Transcription result: {transcription}")
                    
                    # Clean up temporary file
                    try:
                        os.unlink(temp_wav_path)
                    except Exception as e:
                        print(f"Warning: Could not delete temporary file: {e}")
                    
                    return transcription
                else:
                    print("Failed to convert audio file")
                    return None
                
        except Exception as e:
            print(f"Error during audio conversion or transcription: {e}")
            return None
                
    except Exception as e:
        print(f"Error during transcription: {e}")
        return None

def save_description_to_file(description, output_path):
    """Save the description to a text file."""
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(description)
        print(f"Description saved to: {output_path}")
        return True
    except Exception as e:
        print(f"Error saving description to file: {e}")
        return False

def clean_text_for_audio(text):
    """Clean text by removing markdown and special characters for better audio generation."""
    # Remove markdown characters
    text = text.replace('*', '').replace('_', '').replace('**', '')
    # Remove any extra whitespace
    text = ' '.join(text.split())
    return text

def generate_audio_response(text, output_path):
    """Generate audio response from text using gTTS."""
    try:
        # Clean the text before generating audio
        cleaned_text = clean_text_for_audio(text)
        tts = gTTS(text=cleaned_text, lang="vi")
        tts.save(output_path)
        print(f"Audio response saved to: {output_path}")
        return True
    except Exception as e:
        print(f"Error generating audio response: {e}")
        return False

def main():
    # Verify input files exist
    if not os.path.exists(IMAGE_PATH):
        print(f"Error: Image file not found at {IMAGE_PATH}")
        return
        
    if not os.path.exists(AUDIO_PATH):
        print(f"Error: Audio file not found at {AUDIO_PATH}")
        return
    
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Transcribe audio
    information_needs = transcribe_audio(AUDIO_PATH)
    if not information_needs:
        print("Không thể nhận diện giọng nói từ audio.")
        information_needs = "tên, màu sắc, loại sản phẩm và hạn sử dụng"
    
    # Process image with Gemini
    description = process_with_gemini(IMAGE_PATH, information_needs)
    if description:
        print("\nThông tin cần trả lời:")
        print(description)
        
        # Save description to file
        output_txt = os.path.join(OUTPUT_DIR, "output.txt")
        save_description_to_file(description, output_txt)
        
        # Generate and save audio response
        output_audio = os.path.join(OUTPUT_DIR, "output.mp3")
        generate_audio_response(description, output_audio)

if __name__ == "__main__":
    main()
