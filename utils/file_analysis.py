import subprocess
import os
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
try:
    import cv2
    import moviepy.editor as mp
    HAS_VIDEO_SUPPORT = True
except ImportError:
    HAS_VIDEO_SUPPORT = False

try:
    import pytesseract
    from PIL import Image
    HAS_OCR_SUPPORT = True
except ImportError:
    HAS_OCR_SUPPORT = False

def run_command(cmd, input_file):
    """Run a command and return its output."""
    try:
        result = subprocess.run(
            cmd + [str(input_file)],  # Convert Path to string
            capture_output=True,
            text=True,
            timeout=30  # Add timeout
        )
        return result.stdout if result.stdout else result.stderr
    except subprocess.CalledProcessError as e:
        return f"Error running {cmd[0]}: {e.stderr}"
    except subprocess.TimeoutExpired:
        return f"Timeout running {cmd[0]}"
    except Exception as e:
        return f"Error: {str(e)}"

def get_file_metadata(file_path):
    """Extract file metadata using exiftool."""
    try:
        output = run_command(['exiftool'], file_path)
        metadata = {}
        for line in output.split('\n'):
            if ':' in line:
                key, value = line.split(':', 1)
                metadata[key.strip()] = value.strip()
        return metadata
    except Exception as e:
        return {"Error": str(e)}

def extract_strings(file_path, min_length=4):
    """Extract readable strings from the file."""
    try:
        output = run_command(['strings', '-n', str(min_length)], file_path)
        return output.split('\n')
    except Exception as e:
        return [f"Error extracting strings: {str(e)}"]

def analyze_file_structure(file_path):
    """Analyze file structure using binwalk."""
    try:
        output = run_command(['binwalk'], file_path)
        return output
    except Exception as e:
        return f"Error analyzing file structure: {str(e)}"

def calculate_entropy(file_path):
    """Calculate byte-level entropy of the file."""
    try:
        with open(file_path, 'rb') as f:
            data = f.read()

        if len(data) == 0:
            return 0

        entropy = 0
        for x in range(256):
            p_x = data.count(x)/len(data)
            if p_x > 0:
                entropy += -p_x * np.log2(p_x)
        return entropy
    except Exception as e:
        return 0

def get_byte_frequency(file_path):
    """Get byte frequency distribution."""
    try:
        with open(file_path, 'rb') as f:
            data = f.read()

        freq = pd.Series(list(data)).value_counts()
        return freq.index.tolist(), freq.values.tolist()
    except Exception as e:
        return list(range(256)), [0] * 256

def get_hex_dump(file_path, num_bytes=256):
    """Get hexadecimal dump of the file."""
    try:
        output = run_command(['xxd', '-l', str(num_bytes)], file_path)
        return output
    except Exception as e:
        return f"Error getting hex dump: {str(e)}"
        
def run_zsteg(file_path):
    """Run zsteg with -a option on PNG files."""
    try:
        # Check if file is PNG first
        with open(file_path, 'rb') as f:
            header = f.read(8)
            if not header.startswith(b'\x89PNG\r\n\x1a\n'):
                return "ZSTEG only works with PNG files. File format not supported."
        
        # Set up environment with gem bin path
        env = os.environ.copy()
        gem_bin_path = "/home/runner/workspace/.local/share/gem/ruby/3.1.0/bin"
        
        # Add gem bin to PATH if not already there
        if 'PATH' in env:
            if gem_bin_path not in env['PATH']:
                env['PATH'] = f"{env['PATH']}:{gem_bin_path}"
        else:
            env['PATH'] = gem_bin_path
        
        # Try to run zsteg with full path first
        zsteg_cmd = f"{gem_bin_path}/zsteg"
        
        # Check if zsteg exists
        if not os.path.exists(zsteg_cmd):
            return f"ZSTEG not found at {zsteg_cmd}. Please install with: gem install zsteg"
        
        # Run zsteg with various analysis options
        cmd = [zsteg_cmd, "-a", str(file_path)]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
            env=env
        )
        
        output = result.stdout if result.stdout else result.stderr
        
        if not output or output.strip() == "":
            return "No output from ZSTEG analysis"
        
        # Clean up output and filter meaningful results
        lines = output.split('\n')
        meaningful_lines = []
        
        for line in lines:
            line = line.strip()
            # Skip empty lines and common noise
            if not line:
                continue
            # Skip lines that end with ".." (nothing found indicator)
            if line.endswith('..'):
                continue
            # Skip common warnings and non-findings
            if any(skip in line.lower() for skip in [
                'system temporary path', 'world-writable', 'nothing', 
                'possible image block size', 'downscaling may be necessary',
                '[=] nothing'
            ]):
                continue
            meaningful_lines.append(line)
        
        if meaningful_lines:
            return "\n".join(meaningful_lines)
        else:
            return "No hidden data detected by ZSTEG analysis"
        
    except subprocess.TimeoutExpired:
        return "ZSTEG analysis timed out after 30 seconds"
    except Exception as e:
        return f"ZSTEG analysis error: {str(e)}"

def is_video_file(file_path):
    """Check if file is a video format."""
    video_extensions = ['.mp4', '.avi', '.mov', '.wmv', '.flv', '.mkv', '.webm']
    return Path(file_path).suffix.lower() in video_extensions

def extract_video_frames(file_path, max_frames=10):
    """Extract frames from video for steganography analysis."""
    if not HAS_VIDEO_SUPPORT:
        return None, "Video analysis requires opencv-python and moviepy libraries"
    
    if not is_video_file(file_path):
        return None, "File is not a video format"
    
    try:
        # Use OpenCV to extract frames
        if not HAS_VIDEO_SUPPORT:
            return None, "Video support not available"
        cap = cv2.VideoCapture(str(file_path))
        frames = []
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if frame_count == 0:
            cap.release()
            return None, "Unable to read video frames"
        
        # Extract frames evenly distributed throughout the video
        step = max(1, frame_count // max_frames)
        
        for i in range(0, min(frame_count, max_frames * step), step):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if ret:
                frames.append(frame)
            if len(frames) >= max_frames:
                break
        
        cap.release()
        
        if not frames:
            return None, "No frames could be extracted from video"
        
        return frames, f"Extracted {len(frames)} frames for analysis"
        
    except Exception as e:
        return None, f"Error extracting video frames: {str(e)}"

def analyze_video_metadata(file_path):
    """Analyze video metadata for steganography indicators."""
    if not HAS_VIDEO_SUPPORT:
        return {"error": "Video analysis requires moviepy library"}
    
    try:
        # Get basic metadata using exiftool first
        metadata = get_file_metadata(file_path)
        
        # Add video-specific analysis
        try:
            clip = mp.VideoFileClip(str(file_path))
            video_metadata = {
                "duration": clip.duration,
                "fps": clip.fps,
                "size": clip.size,
                "audio_present": clip.audio is not None
            }
            metadata.update(video_metadata)
            clip.close()
        except Exception as e:
            metadata["video_analysis_error"] = str(e)
        
        return metadata
        
    except Exception as e:
        return {"error": f"Error analyzing video metadata: {str(e)}"}

def save_video_frame_for_analysis(frame, temp_dir="/tmp"):
    """Save a video frame as temporary image for steganography analysis."""
    try:
        import tempfile
        from PIL import Image
        
        # Create temporary file
        temp_file = tempfile.NamedTemporaryFile(suffix='.png', delete=False, dir=temp_dir)
        temp_path = temp_file.name
        temp_file.close()
        
        # Convert BGR to RGB (OpenCV uses BGR)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL Image and save
        pil_image = Image.fromarray(rgb_frame)
        pil_image.save(temp_path)
        
        return temp_path
        
    except Exception as e:
        return None

def extract_text_with_ocr(file_path):
    """Extract text from image using OCR (Tesseract)."""
    if not HAS_OCR_SUPPORT:
        return {"error": "OCR support requires pytesseract library"}
    
    try:
        # Open image with PIL
        image = Image.open(file_path)
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Extract text using Tesseract
        extracted_text = pytesseract.image_to_string(image)
        
        # Also get detailed data (confidence scores, etc.)
        detailed_data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
        
        # Filter out low-confidence text
        confident_text = []
        confidences = []
        
        for i, conf in enumerate(detailed_data['conf']):
            if int(conf) > 30:  # Only include text with >30% confidence
                text = detailed_data['text'][i].strip()
                if text:
                    confident_text.append(text)
                    confidences.append(int(conf))
        
        result = {
            "raw_text": extracted_text.strip(),
            "confident_text": " ".join(confident_text),
            "word_count": len(confident_text),
            "average_confidence": sum(confidences) / len(confidences) if confidences else 0,
            "confidence_scores": confidences
        }
        
        return result
        
    except Exception as e:
        return {"error": f"OCR extraction failed: {str(e)}"}

def analyze_text_for_steganography(text):
    """Analyze extracted text for steganographic indicators."""
    if not text or len(text.strip()) == 0:
        return {"likelihood": 0.0, "indicators": []}
    
    indicators = []
    likelihood = 0.0
    
    # Check for common steganographic patterns in text
    
    # 1. Binary strings
    binary_pattern = r'\b[01]{8,}\b'
    import re
    binary_matches = re.findall(binary_pattern, text)
    if binary_matches:
        indicators.append(f"Binary sequences detected: {len(binary_matches)} patterns")
        likelihood += 0.3
    
    # 2. Base64-like strings
    base64_pattern = r'\b[A-Za-z0-9+/]{20,}={0,2}\b'
    base64_matches = re.findall(base64_pattern, text)
    if base64_matches:
        indicators.append(f"Base64-like patterns detected: {len(base64_matches)} patterns")
        likelihood += 0.4
    
    # 3. Hexadecimal strings
    hex_pattern = r'\b[0-9A-Fa-f]{16,}\b'
    hex_matches = re.findall(hex_pattern, text)
    if hex_matches:
        indicators.append(f"Hexadecimal sequences detected: {len(hex_matches)} patterns")
        likelihood += 0.3
    
    # 4. Unusual character frequency
    text_clean = ''.join(c for c in text if c.isalnum())
    if len(text_clean) > 50:
        char_freq = {}
        for char in text_clean.lower():
            char_freq[char] = char_freq.get(char, 0) + 1
        
        # Calculate character entropy
        total_chars = len(text_clean)
        entropy = 0
        for count in char_freq.values():
            prob = count / total_chars
            entropy += -prob * np.log2(prob)
        
        if entropy > 4.0:  # High entropy suggests encoded data
            indicators.append(f"High character entropy: {entropy:.2f}")
            likelihood += 0.2
    
    # 5. Repeated patterns
    words = text.split()
    if len(words) > 10:
        word_freq = {}
        for word in words:
            if len(word) > 3:
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # Check for unusual repetition
        max_freq = max(word_freq.values()) if word_freq else 0
        if max_freq > len(words) * 0.3:  # Word appears >30% of the time
            indicators.append("Unusual word repetition detected")
            likelihood += 0.2
    
    return {
        "likelihood": min(likelihood, 1.0),
        "indicators": indicators,
        "text_length": len(text),
        "clean_length": len(text_clean) if 'text_clean' in locals() else 0
    }