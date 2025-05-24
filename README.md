# Audio Processing and Image Recognition System

This system processes audio files containing questions about products and generates audio responses using Gemini's API for image recognition and gTTS for text-to-speech conversion.

## Features

- Audio file transcription using Whisper model
- Image recognition using Google's Gemini API
- Text-to-speech conversion using gTTS
- Support for various audio formats (MP3, WAV, M4A)
- Vietnamese language support

## Requirements

- Python 3.8 or higher
- Google API key for Gemini
- FFmpeg (for audio processing)

## Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd <repository-name>
```

2. Install the required packages:
```bash
pip install google-generativeai
pip install gtts
pip install Pillow
pip install openai-whisper
pip install pydub
```

3. Install FFmpeg:
   - Download from: https://github.com/BtbN/FFmpeg-Builds/releases
   - Look for "ffmpeg-master-latest-win64-gpl.zip"
   - Extract the zip file
   - Copy the contents of the `bin` folder to a permanent location (e.g., `E:\ffmpeg\bin`)
   - Add FFmpeg to your system PATH:
     1. Open System Properties (Win + Pause/Break)
     2. Click "Advanced system settings"
     3. Click "Environment Variables"
     4. Under "System variables", find and select "Path"
     5. Click "Edit"
     6. Click "New"
     7. Add the path to your FFmpeg bin folder
     8. Click "OK" on all windows

## Configuration

1. Open `main.py` and update the following configuration variables:
```python
GOOGLE_API_KEY = 'YOUR_API_KEY'  # Replace with your Google API key
OUTPUT_DIR = r"path/to/output/directory"  # Replace with your desired output directory
IMAGE_PATH = r"path/to/your/image.jpg"  # Replace with your image path
AUDIO_PATH = r"path/to/your/audio.m4a"  # Replace with your audio path
```

2. Update the FFmpeg path in `main.py`:
```python
FFMPEG_PATH = r"path/to/your/ffmpeg/bin"  # Replace with your FFmpeg bin directory
```

## Usage

1. Prepare your files:
   - Place your image file (JPEG, PNG) in the specified IMAGE_PATH
   - Place your audio file (MP3, WAV, M4A) in the specified AUDIO_PATH

2. Run the script:
```bash
python main.py
```

## Output

The script will generate:
1. `output.txt`: Contains the product description in Vietnamese
2. `output.mp3`: Contains the audio response in Vietnamese
3. Console output showing:
   - FFmpeg installation status
   - Audio conversion progress
   - Transcription results
   - Product description

## Example

```python
# Example configuration in main.py
GOOGLE_API_KEY = 'your-api-key-here'
OUTPUT_DIR = r"C:\Users\YourName\Documents\AI4LI"
IMAGE_PATH = r"C:\Users\YourName\Documents\AI4LI\product.jpg"
AUDIO_PATH = r"C:\Users\YourName\Documents\AI4LI\question.m4a"
FFMPEG_PATH = r"E:\ffmpeg\bin"
```

## Troubleshooting

1. FFmpeg not found:
   - Verify FFmpeg is installed correctly
   - Check if the FFMPEG_PATH in main.py is correct
   - Ensure FFmpeg is in your system PATH

2. Audio conversion issues:
   - Check if the audio file exists and is accessible
   - Verify the audio format is supported
   - Ensure FFmpeg is properly installed

3. Transcription issues:
   - Check if the audio file is clear and in Vietnamese
   - Verify the Whisper model is loaded correctly
   - Ensure sufficient system memory is available

## Notes

- The system is optimized for Vietnamese language
- Audio files are automatically converted to WAV format for processing
- Temporary files are automatically cleaned up after processing
- The Gemini API has usage limits, monitor your API key usage 