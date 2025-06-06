# Naviblind API

API for processing images and audio using Gemini AI and Whisper for visually impaired people.

## Features

- Product recognition from images using Gemini AI
- Text-to-Speech conversion
- Speech-to-Text conversion
- Combined image and audio processing

## System Requirements

- Python 3.8 or higher
- FFmpeg (included in the repository)
- Google API Key for Gemini AI

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Be-Tap-Code/miniature-chainsaw.git
cd miniature-chainsaw
```

2. Install required Python packages:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file in the root directory and add your Google API Key:
```
GOOGLE_API_KEY=your_api_key_here
```
**Important:** Do not commit your `.env` file to version control for security reasons. Add it to your `.gitignore` file.

## Directory Structure

```
miniature-chainsaw/
├── app.py                 # Main FastAPI application
├── main.py               # Command-line interface for Windows
├── main_macos.py         # Command-line interface for macOS
├── api_server.py         # Alternative API server implementation
├── .env                  # Environment variables and API keys
├── requirements.txt      # Python dependencies
├── api_output/          # Directory for processing results
│   ├── audio/          # Generated audio files
│   └── text/           # Generated text files
├── uploads/             # Temporary directory for uploaded files
└── ffmpeg-master-latest-win64-gpl-shared/  # Pre-installed FFmpeg
    └── bin/
        ├── ffmpeg.exe
        ├── ffprobe.exe
        └── ffplay.exe
```

## File Descriptions

- `app.py`: Main FastAPI application with all endpoints and processing logic
- `main.py`: Command-line interface version for Windows users
- `main_macos.py`: Command-line interface version for macOS users
- `api_server.py`: Alternative API server implementation with additional features
- `.env`: Configuration file for environment variables and API keys
- `requirements.txt`: List of required Python packages

## Usage

### Option 1: Using the API Server

1. Start the FastAPI server:
```bash
python app.py
```

2. API will be available at: http://localhost:8000

3. Access API documentation at: http://localhost:8000/docs

### Option 2: Using Command Line Interface

For Windows:
```bash
python main.py
```

For macOS:
```bash
python main_macos.py
```

### Option 3: Using Alternative API Server

```bash
python api_server.py
```

## Main Endpoints

- `GET /`: API information
- `POST /process-image`: Process images
- `POST /transcribe-audio`: Convert audio to text
- `POST /process-combined`: Combined image and audio processing
- `GET /health`: System status check

## Getting Google API Key

1. Visit https://makersuite.google.com/app/apikey
2. Create a new API key
3. Add the API key to your `.env` file

## Platform-Specific Notes

### Windows
- Use `main.py` for command-line interface
- FFmpeg is pre-installed in the repository
- Paths are configured for Windows environment

### macOS
- Use `main_macos.py` for command-line interface
- FFmpeg paths are configured for macOS
- May need to install FFmpeg via Homebrew if not using included version

## Troubleshooting

If you encounter issues, consider the following:
- **API Key:** Ensure your `GOOGLE_API_KEY` in the `.env` file is correct and has the necessary permissions.
- **FFmpeg:** Verify that FFmpeg is correctly set up and accessible in your system's PATH, or that the included FFmpeg binaries are correctly located and configured.

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## License

MIT License
