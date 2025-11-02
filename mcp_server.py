#!/usr/bin/env python3
"""
MCP Server for DEEP ANAL Steganography Analysis
Exposes steganography analysis tools to Claude and other MCP clients
"""

import asyncio
import tempfile
import os
import base64
from pathlib import Path
from typing import Optional
from mcp.server.fastmcp import FastMCP

from utils.file_analysis import (
    get_file_metadata, calculate_entropy, extract_strings,
    run_zsteg
)
from utils.file_identifier import identify_file_type, is_safe_to_analyze
from utils.stego_detector import analyze_image_for_steganography
from utils.stego_decoder import brute_force_decode

mcp = FastMCP("DEEP ANAL Steganography Analyzer")


@mcp.tool()
def analyze_image_steganography(image_path: str) -> dict:
    """
    Perform comprehensive steganography analysis on an image file.
    
    Args:
        image_path: Path to the image file to analyze
        
    Returns:
        Complete analysis including likelihood score, indicators, metadata, and entropy
    """
    try:
        # Verify file exists and is safe
        if not os.path.exists(image_path):
            return {"error": f"File not found: {image_path}"}
        
        file_info = identify_file_type(image_path)
        if not is_safe_to_analyze(image_path):
            return {"error": "File is not safe to analyze (encrypted, executable, or too large)"}
        
        # Run analysis
        detection_result = analyze_image_for_steganography(image_path)
        metadata = get_file_metadata(image_path)
        entropy = calculate_entropy(image_path)
        
        # Prepare response
        result = {
            "file_type": file_info.get("mime_type", "unknown"),
            "file_size": os.path.getsize(image_path),
            "likelihood": detection_result.likelihood,
            "confidence_level": detection_result.confidence_level,
            "indicators": detection_result.indicators,
            "suspected_techniques": detection_result.suspected_techniques,
            "entropy": entropy,
            "metadata_summary": {
                "total_fields": len(metadata),
                "has_exif": any("exif" in k.lower() for k in metadata.keys()),
                "has_comment": any("comment" in k.lower() for k in metadata.keys())
            }
        }
        
        return result
        
    except Exception as e:
        return {"error": f"Analysis failed: {str(e)}"}


@mcp.tool()
def quick_scan_image(image_path: str) -> dict:
    """
    Perform a quick steganography detection scan on an image.
    
    Args:
        image_path: Path to the image file to scan
        
    Returns:
        Quick detection result with likelihood score and basic indicators
    """
    try:
        if not os.path.exists(image_path):
            return {"error": f"File not found: {image_path}"}
        
        detection_result = analyze_image_for_steganography(image_path)
        
        return {
            "likelihood": detection_result.likelihood,
            "confidence_level": detection_result.confidence_level,
            "suspicious": detection_result.likelihood >= 0.3,
            "key_indicators": detection_result.indicators[:5]  # Top 5 indicators
        }
        
    except Exception as e:
        return {"error": f"Scan failed: {str(e)}"}


@mcp.tool()
def extract_hidden_content(image_path: str, max_attempts: int = 10) -> dict:
    """
    Attempt to extract hidden content from an image using multiple techniques.
    
    Args:
        image_path: Path to the image file
        max_attempts: Maximum number of extraction methods to try (default 10)
        
    Returns:
        Extraction results including method, confidence, and extracted content
    """
    try:
        if not os.path.exists(image_path):
            return {"error": f"File not found: {image_path}"}
        
        # Run brute force extraction
        results = brute_force_decode(image_path)
        
        # Filter successful results
        successful = [r for r in results if r.success and r.confidence > 0.2]
        successful = successful[:max_attempts]
        
        extractions = []
        for result in successful:
            extraction = {
                "method": result.method,
                "confidence": result.confidence,
                "data_size": len(result.data) if result.data else 0
            }
            
            # Try to decode as text
            if result.data:
                try:
                    text = result.data.decode('utf-8', errors='ignore')
                    if len(text.strip()) > 0 and all(ord(c) < 127 for c in text[:100]):
                        extraction["content_type"] = "text"
                        extraction["content"] = text[:500]  # Limit for safety
                    else:
                        extraction["content_type"] = "binary"
                        extraction["content_preview"] = ' '.join(f'{b:02x}' for b in result.data[:32])
                except:
                    extraction["content_type"] = "binary"
            
            extractions.append(extraction)
        
        return {
            "total_attempts": len(results),
            "successful_extractions": len(extractions),
            "extractions": extractions
        }
        
    except Exception as e:
        return {"error": f"Extraction failed: {str(e)}"}


@mcp.tool()
def get_file_entropy_analysis(image_path: str) -> dict:
    """
    Calculate entropy and frequency analysis for a file.
    
    Args:
        image_path: Path to the file to analyze
        
    Returns:
        Entropy score and interpretation
    """
    try:
        if not os.path.exists(image_path):
            return {"error": f"File not found: {image_path}"}
        
        entropy = calculate_entropy(image_path)
        
        # Interpret entropy
        if entropy > 7.9:
            interpretation = "Very high entropy - likely encrypted or compressed data"
        elif entropy > 7.5:
            interpretation = "High entropy - possibly contains hidden data or compression"
        elif entropy > 6.0:
            interpretation = "Moderate entropy - normal for image files"
        else:
            interpretation = "Low entropy - simple or repetitive patterns"
        
        return {
            "entropy": entropy,
            "interpretation": interpretation,
            "max_entropy": 8.0
        }
        
    except Exception as e:
        return {"error": f"Entropy analysis failed: {str(e)}"}


@mcp.tool()
def extract_strings_from_file(file_path: str, min_length: int = 4) -> dict:
    """
    Extract readable strings from a file for analysis.
    
    Args:
        file_path: Path to the file
        min_length: Minimum string length to extract (default 4)
        
    Returns:
        Extracted strings and statistics
    """
    try:
        if not os.path.exists(file_path):
            return {"error": f"File not found: {file_path}"}
        
        strings = extract_strings(file_path, min_length=min_length)
        
        # Analyze strings
        urls = [s for s in strings if 'http://' in s or 'https://' in s]
        emails = [s for s in strings if '@' in s and '.' in s]
        suspicious_keywords = [s for s in strings if any(kw in s.lower() for kw in ['password', 'key', 'secret', 'hidden', 'flag'])]
        
        return {
            "total_strings": len(strings),
            "sample_strings": strings[:20],  # First 20 strings
            "urls_found": len(urls),
            "emails_found": len(emails),
            "suspicious_strings": suspicious_keywords[:10],
            "longest_string_length": max((len(s) for s in strings), default=0)
        }
        
    except Exception as e:
        return {"error": f"String extraction failed: {str(e)}"}


@mcp.resource("analysis://help")
def get_help() -> str:
    """Get help information about steganography analysis tools"""
    return """
DEEP ANAL Steganography Analysis Tools:

Available Tools:
1. analyze_image_steganography - Full comprehensive analysis
2. quick_scan_image - Fast detection scan
3. extract_hidden_content - Extract hidden data
4. get_file_entropy_analysis - Entropy calculation
5. extract_strings_from_file - String extraction

Typical Workflow:
1. Start with quick_scan_image to check if image is suspicious
2. If suspicious, run analyze_image_steganography for detailed analysis
3. Use extract_hidden_content to attempt data extraction
4. Use extract_strings_from_file to find readable text

Supported Formats: PNG, JPEG, TIFF, BMP, WEBP, GIF, HEIC/HEIF
"""


if __name__ == "__main__":
    # Run the MCP server
    mcp.run()
