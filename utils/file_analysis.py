import subprocess
import os
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd

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
            if line and not any(skip in line.lower() for skip in [
                'system temporary path', 'world-writable', 'nothing', 
                'possible image block size', 'downscaling may be necessary'
            ]):
                meaningful_lines.append(line)
        
        if meaningful_lines:
            return "\n".join(meaningful_lines)
        else:
            return "No hidden data detected by ZSTEG analysis"
        
    except subprocess.TimeoutExpired:
        return "ZSTEG analysis timed out after 30 seconds"
    except Exception as e:
        return f"ZSTEG analysis error: {str(e)}"