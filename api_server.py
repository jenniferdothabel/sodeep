"""
API Server for ChatGPT Action Integration
Provides REST API endpoints for steganography analysis that can be called from ChatGPT.
"""

import os
import tempfile
import base64
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from pathlib import Path
import json

# Import our analysis modules
from utils.file_analysis import (
    get_file_metadata, calculate_entropy, extract_strings, 
    analyze_file_structure, run_zsteg, extract_text_with_ocr, analyze_text_for_steganography
)
from utils.stego_detector import analyze_image_for_steganography
from utils.stego_decoder import brute_force_decode, extract_with_xor_analysis
from utils.ai_assistant import SteganographyAssistant

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Initialize AI assistant
ai_assistant = SteganographyAssistant()

@app.route('/api/analyze', methods=['POST'])
def analyze_image():
    """
    Analyze an image for steganography via API.
    Accepts either file upload or base64 encoded image.
    """
    try:
        temp_path = None
        
        # Handle different input formats
        if 'file' in request.files:
            # File upload
            file = request.files['file']
            if file.filename == '':
                return jsonify({"error": "No file provided"}), 400
            
            filename = secure_filename(file.filename)
            if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                return jsonify({"error": "Only PNG and JPEG files are supported"}), 400
            
            # Save to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(filename).suffix) as tmp_file:
                file.save(tmp_file.name)
                temp_path = tmp_file.name
                
        elif 'image_base64' in request.json:
            # Base64 encoded image
            try:
                image_data = base64.b64decode(request.json['image_base64'])
                with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_file:
                    tmp_file.write(image_data)
                    temp_path = tmp_file.name
                filename = request.json.get('filename', 'uploaded_image.png')
            except Exception as e:
                return jsonify({"error": f"Invalid base64 image data: {str(e)}"}), 400
        else:
            return jsonify({"error": "No image data provided"}), 400
        
        # Perform analysis
        try:
            # Basic file analysis
            file_size = os.path.getsize(temp_path)
            entropy = calculate_entropy(temp_path)
            metadata = get_file_metadata(temp_path)
            
            # Steganography detection
            detection_result = analyze_image_for_steganography(temp_path)
            likelihood = detection_result.likelihood
            
            # If likelihood is high enough, try extraction
            extracted_content = None
            extraction_results = []
            
            if likelihood >= 0.3:  # Lower threshold for API
                try:
                    results = brute_force_decode(temp_path)
                    successful_results = [r for r in results if r.success and r.confidence > 0.2]
                    
                    for result in successful_results[:3]:  # Top 3 results
                        extraction_data = {
                            "method": result.method,
                            "confidence": result.confidence,
                            "success": result.success
                        }
                        
                        # Handle extracted data
                        if result.data:
                            try:
                                # Try to decode as text
                                text_data = result.data.decode('utf-8', errors='ignore')
                                if len(text_data.strip()) > 0 and all(ord(c) < 127 for c in text_data[:100]):
                                    extraction_data["content_type"] = "text"
                                    extraction_data["content"] = text_data[:1000]  # Limit for API
                                else:
                                    extraction_data["content_type"] = "binary"
                                    extraction_data["content_size"] = len(result.data)
                                    extraction_data["content_preview"] = ' '.join(f'{b:02x}' for b in result.data[:32])
                            except:
                                extraction_data["content_type"] = "binary"
                                extraction_data["content_size"] = len(result.data) if result.data else 0
                        
                        extraction_results.append(extraction_data)
                        
                        # Store best extraction for AI analysis
                        if not extracted_content and result.data:
                            extracted_content = result.data
                            
                except Exception as e:
                    extraction_results.append({"error": f"Extraction failed: {str(e)}"})
            
            # Get AI analysis
            ai_analysis = ai_assistant.analyze_detection_results(
                detection_result, metadata, extracted_content
            )
            
            # Prepare response
            response_data = {
                "analysis_summary": {
                    "filename": filename,
                    "file_size": file_size,
                    "entropy": entropy,
                    "steganography_likelihood": likelihood,
                    "likelihood_percentage": f"{likelihood*100:.1f}%",
                    "risk_level": "high" if likelihood >= 0.7 else "medium" if likelihood >= 0.4 else "low"
                },
                "detection_details": {
                    "indicators": detection_result.indicators if hasattr(detection_result, 'indicators') else {},
                    "explanation": detection_result.explanation if hasattr(detection_result, 'explanation') else "",
                    "techniques": detection_result.techniques if hasattr(detection_result, 'techniques') else []
                },
                "extracted_content": extraction_results,
                "ai_insights": ai_analysis,
                "recommendations": ai_analysis.get("investigation_recommendations", []),
                "metadata_summary": {
                    "total_fields": len(metadata),
                    "suspicious_fields": [k for k in metadata.keys() if any(term in k.lower() for term in ['comment', 'description', 'author', 'copyright'])]
                }
            }
            
            return jsonify(response_data)
            
        finally:
            # Cleanup temporary file
            if temp_path and os.path.exists(temp_path):
                os.unlink(temp_path)
                
    except Exception as e:
        # Cleanup on error
        if 'temp_path' in locals() and temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)
        return jsonify({"error": f"Analysis failed: {str(e)}"}), 500

@app.route('/api/quick-scan', methods=['POST'])
def quick_scan():
    """
    Quick steganography scan - detection only, no extraction.
    Faster for initial assessment.
    """
    try:
        temp_path = None
        
        # Handle file input (same as analyze_image)
        if 'file' in request.files:
            file = request.files['file']
            if file.filename == '':
                return jsonify({"error": "No file provided"}), 400
                
            filename = secure_filename(file.filename)
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(filename).suffix) as tmp_file:
                file.save(tmp_file.name)
                temp_path = tmp_file.name
        elif 'image_base64' in request.json:
            image_data = base64.b64decode(request.json['image_base64'])
            with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_file:
                tmp_file.write(image_data)
                temp_path = tmp_file.name
            filename = request.json.get('filename', 'uploaded_image.png')
        else:
            return jsonify({"error": "No image data provided"}), 400
        
        try:
            # Quick analysis
            file_size = os.path.getsize(temp_path)
            entropy = calculate_entropy(temp_path)
            detection_result = analyze_image_for_steganography(temp_path)
            
            response_data = {
                "filename": filename,
                "file_size": file_size,
                "entropy": entropy,
                "steganography_likelihood": detection_result.likelihood,
                "likelihood_percentage": f"{detection_result.likelihood*100:.1f}%",
                "risk_assessment": "high" if detection_result.likelihood >= 0.7 else "medium" if detection_result.likelihood >= 0.4 else "low",
                "explanation": detection_result.explanation if hasattr(detection_result, 'explanation') else "",
                "recommended_action": "immediate_extraction" if detection_result.likelihood >= 0.7 else "detailed_analysis" if detection_result.likelihood >= 0.4 else "monitor"
            }
            
            return jsonify(response_data)
            
        finally:
            if temp_path and os.path.exists(temp_path):
                os.unlink(temp_path)
                
    except Exception as e:
        if 'temp_path' in locals() and temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)
        return jsonify({"error": f"Quick scan failed: {str(e)}"}), 500

@app.route('/api/ocr-extract', methods=['POST'])
def ocr_extract():
    """Extract text from images using OCR and analyze for steganographic patterns."""
    try:
        temp_path = None
        
        # Handle file input
        if 'file' in request.files:
            file = request.files['file']
            if file.filename == '':
                return jsonify({"error": "No file provided"}), 400
                
            filename = secure_filename(file.filename)
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(filename).suffix) as tmp_file:
                file.save(tmp_file.name)
                temp_path = tmp_file.name
        elif 'image_base64' in request.json:
            image_data = base64.b64decode(request.json['image_base64'])
            with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_file:
                tmp_file.write(image_data)
                temp_path = tmp_file.name
            filename = request.json.get('filename', 'uploaded_image.png')
        else:
            return jsonify({"error": "No image data provided"}), 400
        
        try:
            # Perform OCR extraction
            ocr_result = extract_text_with_ocr(temp_path)
            
            if "error" not in ocr_result:
                # Analyze text for steganographic patterns
                text_analysis = None
                if ocr_result['raw_text']:
                    text_analysis = analyze_text_for_steganography(ocr_result['raw_text'])
                
                response_data = {
                    "filename": filename,
                    "ocr_results": {
                        "word_count": ocr_result['word_count'],
                        "average_confidence": ocr_result['average_confidence'],
                        "raw_text": ocr_result['raw_text'][:2000] if ocr_result['raw_text'] else None,  # Limit for API
                        "text_length": len(ocr_result['raw_text']) if ocr_result['raw_text'] else 0
                    },
                    "steganography_analysis": text_analysis if text_analysis else None,
                    "suspicious_patterns": text_analysis['likelihood'] > 0.3 if text_analysis else False,
                    "success": True
                }
                
                return jsonify(response_data)
            else:
                return jsonify({
                    "filename": filename,
                    "success": False,
                    "error": ocr_result['error']
                }), 400
                
        finally:
            if temp_path and os.path.exists(temp_path):
                os.unlink(temp_path)
                
    except Exception as e:
        if 'temp_path' in locals() and temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)
        return jsonify({"error": f"OCR extraction failed: {str(e)}"}), 500

@app.route('/api/xor-decode', methods=['POST'])
def xor_decode():
    """Perform XOR analysis on images to find hidden data."""
    try:
        temp_path = None
        
        # Handle file input
        if 'file' in request.files:
            file = request.files['file']
            if file.filename == '':
                return jsonify({"error": "No file provided"}), 400
                
            filename = secure_filename(file.filename)
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(filename).suffix) as tmp_file:
                file.save(tmp_file.name)
                temp_path = tmp_file.name
        elif 'image_base64' in request.json:
            image_data = base64.b64decode(request.json['image_base64'])
            with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_file:
                tmp_file.write(image_data)
                temp_path = tmp_file.name
            filename = request.json.get('filename', 'uploaded_image.png')
        else:
            return jsonify({"error": "No image data provided"}), 400
        
        try:
            # Perform XOR analysis
            xor_results = extract_with_xor_analysis(temp_path)
            
            if xor_results:
                successful_results = [r for r in xor_results if r.success and r.confidence > 0.3]
                
                # Format results for API response
                formatted_results = []
                for result in successful_results[:5]:  # Top 5 results
                    result_data = {
                        "method": result.method,
                        "confidence": result.confidence,
                        "success": result.success
                    }
                    
                    if result.data:
                        try:
                            # Try to decode as text
                            text_data = result.data.decode('utf-8', errors='ignore')
                            if text_data.strip() and len([c for c in text_data if c.isprintable()]) / len(text_data) > 0.7:
                                result_data["content_type"] = "text"
                                result_data["content"] = text_data[:1000]  # Limit for API
                            else:
                                result_data["content_type"] = "binary"
                                result_data["content_size"] = len(result.data)
                                result_data["content_preview"] = result.data[:32].hex()
                        except:
                            result_data["content_type"] = "binary"
                            result_data["content_size"] = len(result.data) if result.data else 0
                            result_data["content_preview"] = result.data[:32].hex() if result.data else ""
                    
                    formatted_results.append(result_data)
                
                response_data = {
                    "filename": filename,
                    "total_results": len(xor_results),
                    "successful_results": len(successful_results),
                    "xor_decoded_content": formatted_results,
                    "analysis_summary": {
                        "best_confidence": max([r.confidence for r in successful_results]) if successful_results else 0,
                        "methods_tried": len(xor_results),
                        "potential_hidden_data": len(successful_results) > 0
                    },
                    "success": True
                }
                
                return jsonify(response_data)
            else:
                return jsonify({
                    "filename": filename,
                    "total_results": 0,
                    "successful_results": 0,
                    "analysis_summary": {
                        "potential_hidden_data": False,
                        "methods_tried": 0
                    },
                    "success": True,
                    "message": "No XOR patterns detected"
                })
                
        finally:
            if temp_path and os.path.exists(temp_path):
                os.unlink(temp_path)
                
    except Exception as e:
        if 'temp_path' in locals() and temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)
        return jsonify({"error": f"XOR analysis failed: {str(e)}"}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint for monitoring."""
    return jsonify({
        "status": "healthy",
        "service": "DEEP ANAL Steganography API",
        "version": "1.0",
        "capabilities": ["image_analysis", "steganography_detection", "content_extraction", "ai_analysis", "ocr_extraction", "xor_decoding"]
    })

@app.route('/.well-known/ai-plugin.json', methods=['GET'])
def ai_plugin_manifest():
    """ChatGPT action plugin manifest."""
    return jsonify({
        "schema_version": "v1",
        "name_for_model": "steganography_analyzer",
        "name_for_human": "DEEP ANAL Steganography Analyzer",
        "description_for_model": "Analyze images for hidden data using advanced steganography detection. Can detect LSB steganography, metadata hiding, extract hidden content, perform OCR text extraction with pattern analysis, and XOR decoding from PNG and JPEG images.",
        "description_for_human": "Advanced steganography analysis tool that can detect and extract hidden data from images.",
        "auth": {
            "type": "none"
        },
        "api": {
            "type": "openapi",
            "url": f"{request.host_url}openapi.json"
        },
        "logo_url": f"{request.host_url}static/logo.png",
        "contact_email": "support@deepanal.ai",
        "legal_info_url": f"{request.host_url}legal"
    })

@app.route('/openapi.json', methods=['GET'])
def openapi_spec():
    """OpenAPI specification for ChatGPT action."""
    return jsonify({
        "openapi": "3.0.0",
        "info": {
            "title": "DEEP ANAL Steganography API",
            "description": "Advanced steganography analysis and hidden content extraction API",
            "version": "1.0.0"
        },
        "servers": [
            {
                "url": request.host_url.rstrip('/'),
                "description": "DEEP ANAL Steganography Analysis Server"
            }
        ],
        "paths": {
            "/api/analyze": {
                "post": {
                    "summary": "Comprehensive steganography analysis",
                    "description": "Performs complete steganography analysis including detection, extraction, and AI-powered insights",
                    "operationId": "analyzeImage",
                    "requestBody": {
                        "required": True,
                        "content": {
                            "multipart/form-data": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "file": {
                                            "type": "string",
                                            "format": "binary",
                                            "description": "Image file to analyze (PNG or JPEG)"
                                        }
                                    },
                                    "required": ["file"]
                                }
                            },
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "image_base64": {
                                            "type": "string",
                                            "description": "Base64 encoded image data"
                                        },
                                        "filename": {
                                            "type": "string",
                                            "description": "Original filename"
                                        }
                                    },
                                    "required": ["image_base64"]
                                }
                            }
                        }
                    },
                    "responses": {
                        "200": {
                            "description": "Analysis completed successfully",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "analysis_summary": {
                                                "type": "object",
                                                "description": "Summary of the analysis results"
                                            },
                                            "detection_details": {
                                                "type": "object", 
                                                "description": "Detailed steganography detection results"
                                            },
                                            "extracted_content": {
                                                "type": "array",
                                                "description": "Any hidden content found in the image"
                                            },
                                            "ai_insights": {
                                                "type": "object",
                                                "description": "AI-powered analysis and recommendations"
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            },
            "/api/quick-scan": {
                "post": {
                    "summary": "Quick steganography detection scan",
                    "description": "Fast detection-only scan to check if an image likely contains hidden data",
                    "operationId": "quickScan",
                    "requestBody": {
                        "required": True,
                        "content": {
                            "multipart/form-data": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "file": {
                                            "type": "string",
                                            "format": "binary"
                                        }
                                    }
                                }
                            }
                        }
                    },
                    "responses": {
                        "200": {
                            "description": "Quick scan completed",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "steganography_likelihood": {
                                                "type": "number",
                                                "description": "Probability of hidden content (0-1)"
                                            },
                                            "risk_assessment": {
                                                "type": "string",
                                                "description": "Risk level: low, medium, or high"
                                            },
                                            "recommended_action": {
                                                "type": "string",
                                                "description": "Suggested next step"
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            },
            "/api/ocr-extract": {
                "post": {
                    "summary": "OCR text extraction with steganographic analysis",
                    "description": "Extract text from images using OCR and analyze for steganographic patterns",
                    "operationId": "ocrExtractText",
                    "requestBody": {
                        "required": True,
                        "content": {
                            "multipart/form-data": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "file": {
                                            "type": "string",
                                            "format": "binary",
                                            "description": "Image file for OCR processing"
                                        }
                                    },
                                    "required": ["file"]
                                }
                            }
                        }
                    },
                    "responses": {
                        "200": {
                            "description": "OCR extraction completed successfully",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "filename": {"type": "string"},
                                            "ocr_results": {
                                                "type": "object",
                                                "properties": {
                                                    "word_count": {"type": "integer"},
                                                    "average_confidence": {"type": "number"},
                                                    "raw_text": {"type": "string"},
                                                    "text_length": {"type": "integer"}
                                                }
                                            },
                                            "steganography_analysis": {"type": "object"},
                                            "suspicious_patterns": {"type": "boolean"}
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            },
            "/api/xor-decode": {
                "post": {
                    "summary": "XOR decoding analysis",
                    "description": "Perform XOR analysis to decode potentially hidden data",
                    "operationId": "xorDecodeAnalysis",
                    "requestBody": {
                        "required": True,
                        "content": {
                            "multipart/form-data": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "file": {
                                            "type": "string",
                                            "format": "binary",
                                            "description": "Image file for XOR analysis"
                                        }
                                    },
                                    "required": ["file"]
                                }
                            }
                        }
                    },
                    "responses": {
                        "200": {
                            "description": "XOR analysis completed successfully",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "filename": {"type": "string"},
                                            "total_results": {"type": "integer"},
                                            "successful_results": {"type": "integer"},
                                            "xor_decoded_content": {
                                                "type": "array",
                                                "items": {
                                                    "type": "object",
                                                    "properties": {
                                                        "method": {"type": "string"},
                                                        "confidence": {"type": "number"},
                                                        "content_type": {"type": "string"},
                                                        "content": {"type": "string"}
                                                    }
                                                }
                                            },
                                            "analysis_summary": {"type": "object"}
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5005))
    print(f"Starting DEEP ANAL API server on port {port}")
    try:
        app.run(host='0.0.0.0', port=port, debug=False, threaded=True)
    except Exception as e:
        print(f"Failed to start server: {e}")
        import traceback
        traceback.print_exc()