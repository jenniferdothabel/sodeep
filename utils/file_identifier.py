"""
File Type Identification Module
Uses the 'file' command (libmagic) to identify file types by their binary signatures
"""

import subprocess
import os

def identify_file_type(file_path):
    """
    Identify file type using magic numbers and binary signatures.
    
    Args:
        file_path: Path to the file to identify
        
    Returns:
        dict: {
            'mime_type': 'image/png',
            'description': 'PNG image data, 800 x 600, 8-bit/color RGB, non-interlaced',
            'extension': '.png',
            'category': 'image',
            'is_binary': True,
            'is_executable': False,
            'is_compressed': False,
            'is_encrypted': False
        }
    """
    if not os.path.exists(file_path):
        return {"error": "File not found"}
    
    try:
        # Get MIME type
        mime_result = subprocess.run(
            ['file', '--mime-type', '-b', file_path],
            capture_output=True,
            text=True,
            timeout=5
        )
        mime_type = mime_result.stdout.strip()
        
        # Get detailed description
        desc_result = subprocess.run(
            ['file', '-b', file_path],
            capture_output=True,
            text=True,
            timeout=5
        )
        description = desc_result.stdout.strip()
        
        # Determine category from MIME type
        category = mime_type.split('/')[0] if '/' in mime_type else 'unknown'
        
        # Determine characteristics
        desc_lower = description.lower()
        is_binary = not ('text' in mime_type or 'ascii' in desc_lower)
        is_executable = 'executable' in desc_lower or 'script' in desc_lower
        is_compressed = any(x in desc_lower for x in ['compressed', 'archive', 'zip', 'gzip', 'bzip', 'tar'])
        is_encrypted = 'encrypted' in desc_lower or 'pgp' in desc_lower
        
        # Try to guess extension from MIME type
        extension_map = {
            'image/png': '.png',
            'image/jpeg': '.jpg',
            'image/gif': '.gif',
            'image/webp': '.webp',
            'image/tiff': '.tiff',
            'image/bmp': '.bmp',
            'image/heic': '.heic',
            'video/mp4': '.mp4',
            'video/x-msvideo': '.avi',
            'video/quicktime': '.mov',
            'application/pdf': '.pdf',
            'application/zip': '.zip',
            'text/plain': '.txt',
            'text/html': '.html',
            'application/json': '.json',
            'application/x-executable': '.exe',
            'application/x-dosexec': '.exe',
        }
        extension = extension_map.get(mime_type, '')
        
        return {
            'mime_type': mime_type,
            'description': description,
            'extension': extension,
            'category': category,
            'is_binary': is_binary,
            'is_executable': is_executable,
            'is_compressed': is_compressed,
            'is_encrypted': is_encrypted,
            'raw_output': description
        }
        
    except subprocess.TimeoutExpired:
        return {"error": "File identification timed out"}
    except Exception as e:
        return {"error": f"File identification failed: {str(e)}"}


def get_file_signature(file_path, bytes_to_read=16):
    """
    Read the file signature (magic bytes) from the beginning of a file.
    
    Args:
        file_path: Path to the file
        bytes_to_read: Number of bytes to read (default 16)
        
    Returns:
        dict: {
            'hex': '89504e470d0a1a0a',
            'bytes': b'\x89PNG\r\n\x1a\n',
            'ascii': 'PNG....'
        }
    """
    try:
        with open(file_path, 'rb') as f:
            signature = f.read(bytes_to_read)
            
        return {
            'hex': signature.hex(),
            'bytes': signature,
            'ascii': ''.join(chr(b) if 32 <= b < 127 else '.' for b in signature),
            'length': len(signature)
        }
    except Exception as e:
        return {"error": f"Failed to read file signature: {str(e)}"}


def is_safe_to_analyze(file_path, max_size_mb=100):
    """
    Check if a file is safe to analyze (not executable, not too large).
    
    Args:
        file_path: Path to the file
        max_size_mb: Maximum file size in MB (default 100)
        
    Returns:
        dict: {
            'is_safe': True/False,
            'warnings': [],
            'file_size_mb': 5.2
        }
    """
    warnings = []
    is_safe = True
    
    try:
        # Check file size
        file_size = os.path.getsize(file_path)
        file_size_mb = file_size / (1024 * 1024)
        
        if file_size_mb > max_size_mb:
            warnings.append(f"File is very large ({file_size_mb:.1f} MB)")
            is_safe = False
        
        # Identify file type
        file_info = identify_file_type(file_path)
        
        if "error" not in file_info:
            # Check for executables
            if file_info.get('is_executable'):
                warnings.append("File appears to be executable")
                is_safe = False
            
            # Check for encryption
            if file_info.get('is_encrypted'):
                warnings.append("File appears to be encrypted")
        
        return {
            'is_safe': is_safe,
            'warnings': warnings,
            'file_size_mb': file_size_mb,
            'file_info': file_info if "error" not in file_info else None
        }
        
    except Exception as e:
        return {
            'is_safe': False,
            'warnings': [f"Safety check failed: {str(e)}"],
            'file_size_mb': 0
        }
