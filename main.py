import streamlit as st
import tempfile
import os
import json
from datetime import datetime
from pathlib import Path
import base64
import html
from PIL import Image

# Enable HEIF support
try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
    HEIF_AVAILABLE = True
except ImportError:
    HEIF_AVAILABLE = False
from utils.file_analysis import (
    get_file_metadata, extract_strings, analyze_file_structure,
    calculate_entropy, get_byte_frequency, get_hex_dump, run_zsteg,
    is_video_file, extract_video_frames, analyze_video_metadata, save_video_frame_for_analysis,
    extract_text_with_ocr, analyze_text_for_steganography
)
from utils.visualizations import (
    create_entropy_plot, create_byte_frequency_plot, format_hex_dump,
    create_detailed_view, create_strings_visualization,
    create_channel_analysis_visualization, create_channel_comparison_plot,
    create_byte_frequency_plot_upgraded, create_bitplane_visualizer,
    create_rgb_3d_scatter, create_entropy_terrain_map, create_segment_structure_mapper
)
from utils.database import (
    save_analysis, get_recent_analyses, get_analysis_by_id, DB_AVAILABLE
)
from utils.stego_detector import analyze_image_for_steganography, analyze_binary_file_for_steganography
from utils.stego_decoder import (
    brute_force_decode, decode_lsb, decode_multi_bit_lsb, 
    try_steghide_extract, extract_metadata_hidden_data, extract_with_xor_analysis
)
from utils.file_identifier import identify_file_type, get_file_signature, is_safe_to_analyze
try:
    from utils.ai_assistant import SteganographyAssistant, get_investigation_suggestions
    AI_AVAILABLE = True
except ImportError as e:
    AI_AVAILABLE = False
    SteganographyAssistant = None
    def get_investigation_suggestions(likelihood, indicators):
        return ["AI Assistant not available - limited suggestions"]
        
# App version for cache busting
APP_VERSION = "2.1.0"  # Universal file support update

def load_css():
    """Load cyberpunk CSS styling"""
    with open('.streamlit/style.css') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
        
def create_terminal_panel(title, content, status="online"):
    """Create a cyberpunk terminal-style panel"""
    status_color = "#00ff00" if status == "online" else "#ff0040" if status == "error" else "#ff8c00"
    status_icon = "●" if status == "online" else "⚠" if status == "warning" else "✕"
    
    st.markdown(f"""
    <div class="terminal-window">
        <div class="terminal-header">
            <span style="color: {status_color}">{status_icon}</span> {title.upper()} - STATUS: {status.upper()}
        </div>
        <div class="terminal-body">
            {content}
        </div>
    </div>
    """, unsafe_allow_html=True)
    
def create_holo_panel(content):
    """Create a holographic data visualization panel"""
    st.markdown(f"""
    <div class="holo-panel">
        {content}
    </div>
    """, unsafe_allow_html=True)

def save_extracted_binary(data, method_name, method_index=None):
    """Save extracted binary data to a file for external analysis"""
    try:
        if method_index is None:
            filename = "hidden_payload.bin"
        else:
            filename = f"hidden_payload_method{method_index}.bin"
        
        # Write raw binary data
        with open(filename, "wb") as f:
            f.write(data)
        
        return filename
    except Exception as e:
        return None

def create_bulk_extraction_csv(results, filename_prefix="bulk_extraction"):
    """Create CSV download for multiple extraction results"""
    if not results:
        return None
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_lines = ["Method,Success,Confidence,Content_Type,Size_Bytes,Preview"]
    
    for result in results:
        try:
            method = result.method if hasattr(result, 'method') else 'Unknown'
            success = str(result.success) if hasattr(result, 'success') else 'Unknown'
            confidence = f"{result.confidence:.3f}" if hasattr(result, 'confidence') else '0'
            
            if result.data:
                size_bytes = len(result.data)
                try:
                    # Try to determine if it's text
                    text_data = result.data.decode('utf-8', errors='ignore')
                    if len(text_data.strip()) > 0 and all(ord(c) < 127 for c in text_data[:100]):
                        content_type = "text"
                        preview = text_data[:100].replace('\n', '\\n').replace('\r', '\\r').replace('"', "'")
                    else:
                        content_type = "binary"
                        preview = result.data[:16].hex() + "..." if len(result.data) > 16 else result.data.hex()
                except:
                    content_type = "binary" 
                    preview = result.data[:16].hex() + "..." if len(result.data) > 16 else result.data.hex()
            else:
                size_bytes = 0
                content_type = "empty"
                preview = ""
            
            # Escape CSV values
            csv_line = f'"{method}",{success},{confidence},{content_type},{size_bytes},"{preview}"'
            csv_lines.append(csv_line)
        except Exception as e:
            csv_lines.append(f'"Error",False,0,error,0,"Failed to process: {str(e)}"')
    
    csv_content = '\n'.join(csv_lines)
    filename = f"{filename_prefix}_{timestamp}.csv"
    
    return csv_content, filename

def create_extraction_download_buttons(data, method_name, is_text=True):
    """Create download buttons for extracted content in multiple formats"""
    if not data:
        return
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    # Better filename sanitization
    safe_method = method_name.replace(' ', '_').replace('/', '_').replace('\\', '_').replace(':', '_').replace('"', '_').replace("'", '_')
    
    if is_text:
        # Text content - 4 columns
        col1, col2, col3, col4 = st.columns(4)
        text_data = data if isinstance(data, str) else data.decode('utf-8', errors='ignore')
        binary_data = text_data.encode('utf-8')
        
        with col1:
            st.download_button(
                label="📄 .txt",
                data=text_data,
                file_name=f"extracted_{safe_method}_{timestamp}.txt",
                mime="text/plain",
                help="Download as text file",
                use_container_width=True
            )
        
        with col2:
            # JSON format for text
            json_data = {
                "method": method_name,
                "extraction_timestamp": timestamp,
                "content_type": "text",
                "content": text_data,
                "size_bytes": len(binary_data),
                "encoding": "utf-8"
            }
            st.download_button(
                label="📋 .json",
                data=json.dumps(json_data, indent=2, ensure_ascii=False),
                file_name=f"extracted_{safe_method}_{timestamp}.json",
                mime="application/json",
                help="Download as structured JSON",
                use_container_width=True
            )
        
        with col3:
            # Hex format with size limit
            if len(binary_data) > 50000:  # 50KB limit for hex display
                hex_preview = binary_data[:10000].hex()
                formatted_hex = f"# File too large for full hex display\n# Showing first 10KB of {len(binary_data)} bytes\n" + ' '.join(hex_preview[i:i+2] for i in range(0, len(hex_preview), 2))
            else:
                hex_data = binary_data.hex()
                formatted_hex = ' '.join(hex_data[i:i+2] for i in range(0, len(hex_data), 2))
            
            st.download_button(
                label="🔍 .hex",
                data=formatted_hex,
                file_name=f"extracted_{safe_method}_{timestamp}.hex",
                mime="text/plain",
                help="Download as hexadecimal",
                use_container_width=True
            )
        
        with col4:
            st.download_button(
                label="💾 .bin",
                data=binary_data,
                file_name=f"extracted_{safe_method}_{timestamp}.bin",
                mime="application/octet-stream",
                help="Download as binary file",
                use_container_width=True
            )
    else:
        # Binary content - 3 columns (no .txt)
        col1, col2, col3 = st.columns(3)
        binary_data = data if isinstance(data, bytes) else data.encode('utf-8')
        
        with col1:
            # JSON format for binary with base64 encoding
            import base64
            json_data = {
                "method": method_name,
                "extraction_timestamp": timestamp,
                "content_type": "binary",
                "encoding": "base64",
                "content": base64.b64encode(binary_data).decode('ascii'),
                "size_bytes": len(binary_data),
                "preview_hex": binary_data[:32].hex() if len(binary_data) > 0 else ""
            }
            st.download_button(
                label="📋 .json",
                data=json.dumps(json_data, indent=2),
                file_name=f"extracted_{safe_method}_{timestamp}.json",
                mime="application/json",
                help="Download as structured JSON with base64 content",
                use_container_width=True
            )
        
        with col2:
            # Hex format with size limit
            if len(binary_data) > 50000:  # 50KB limit
                hex_preview = binary_data[:10000].hex()
                formatted_hex = f"# File too large for full hex display\n# Showing first 10KB of {len(binary_data)} bytes\n" + ' '.join(hex_preview[i:i+2] for i in range(0, len(hex_preview), 2))
            else:
                hex_data = binary_data.hex()
                formatted_hex = ' '.join(hex_data[i:i+2] for i in range(0, len(hex_data), 2))
            
            st.download_button(
                label="🔍 .hex",
                data=formatted_hex,
                file_name=f"extracted_{safe_method}_{timestamp}.hex",
                mime="text/plain",
                help="Download as hexadecimal",
                use_container_width=True
            )
        
        with col3:
            st.download_button(
                label="💾 .bin",
                data=binary_data,
                file_name=f"extracted_{safe_method}_{timestamp}.bin",
                mime="application/octet-stream",
                help="Download as binary file",
                use_container_width=True
            )

def generate_detection_report(filename, detection_result, metadata, likelihood):
    """Generate comprehensive detection report for download"""
    
    report = {
        "analysis_metadata": {
            "filename": filename,
            "analysis_timestamp": datetime.now().isoformat(),
            "analysis_version": "DEEP ANAL v1.0",
            "analysis_type": "steganography_detection"
        },
        "detection_summary": {
            "steganography_likelihood": float(likelihood),
            "likelihood_percentage": f"{likelihood * 100:.1f}%",
            "risk_level": "high" if likelihood >= 0.7 else "medium" if likelihood >= 0.3 else "low"
        },
        "detection_details": {},
        "file_metadata": metadata,
        "technical_indicators": {}
    }
    
    # Add detection details if available
    if hasattr(detection_result, 'indicators'):
        report["technical_indicators"] = detection_result.indicators
    
    if hasattr(detection_result, 'explanation'):
        report["detection_details"]["explanation"] = detection_result.explanation
    
    if hasattr(detection_result, 'techniques'):
        report["detection_details"]["suspected_techniques"] = detection_result.techniques
    
    # Add file analysis data
    try:
        if 'File Size' in metadata:
            report["file_analysis"] = {
                "file_size_bytes": metadata.get('File Size', 'Unknown'),
                "image_dimensions": f"{metadata.get('Image Width', 'N/A')} x {metadata.get('Image Height', 'N/A')}",
                "color_space": metadata.get('Color Space', 'Unknown'),
                "compression": metadata.get('Compression', 'Unknown')
            }
    except:
        pass
    
    return report

def generate_comprehensive_html_report(filename, detection_result, metadata, likelihood, extracted_data=None, channel_analysis=None, file_size=0, entropy=0, image_path=None):
    """Generate comprehensive HTML report with all analysis data and interactive visualizations"""
    
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    risk_level = 'HIGH' if likelihood >= 0.7 else 'MEDIUM' if likelihood >= 0.3 else 'LOW'
    risk_color = '#ff0040' if likelihood >= 0.7 else '#ff8c00' if likelihood >= 0.3 else '#00ff00'
    
    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>DEEP ANAL - Analysis Report: {filename}</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js" charset="utf-8"></script>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Share+Tech+Mono&display=swap');
        
        body {{
            font-family: 'Share Tech Mono', monospace;
            background: #000;
            color: #00ff00;
            margin: 0;
            padding: 20px;
            line-height: 1.6;
        }}
        
        @media print {{
            body {{
                background: white;
                color: black;
            }}
            .neon-border {{
                border: 2px solid black !important;
            }}
            .header {{
                background: linear-gradient(45deg, #333, #666) !important;
                -webkit-background-clip: text !important;
                -webkit-text-fill-color: transparent !important;
            }}
        }}
        
        .header {{
            font-family: 'Orbitron', monospace;
            font-weight: 900;
            text-align: center;
            font-size: 2.5rem;
            background: linear-gradient(45deg, #00ffff, #ff00ff, #9400d3);
            background-clip: text;
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-shadow: 0 0 20px #00ffff;
            margin-bottom: 30px;
        }}
        
        .section {{
            margin: 30px 0;
            padding: 20px;
            border: 2px solid #00ffff;
            border-radius: 10px;
            background: rgba(0, 255, 255, 0.05);
        }}
        
        .neon-border {{
            border: 2px solid #ff00ff;
            padding: 15px;
            margin: 15px 0;
            border-radius: 8px;
            background: rgba(255, 0, 255, 0.05);
        }}
        
        .risk-high {{ color: #ff0040; font-weight: bold; }}
        .risk-medium {{ color: #ff8c00; font-weight: bold; }}
        .risk-low {{ color: #00ff00; font-weight: bold; }}
        
        .metric {{
            display: inline-block;
            margin: 10px 20px 10px 0;
            padding: 10px;
            border: 1px solid #9400d3;
            border-radius: 5px;
            background: rgba(148, 0, 211, 0.1);
        }}
        
        .metric-label {{
            color: #9400d3;
            font-size: 0.9rem;
        }}
        
        .metric-value {{
            color: #00ffff;
            font-weight: bold;
            font-size: 1.2rem;
        }}
        
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 15px 0;
        }}
        
        th, td {{
            border: 1px solid #00ffff;
            padding: 8px 12px;
            text-align: left;
        }}
        
        th {{
            background: rgba(0, 255, 255, 0.2);
            color: #00ffff;
            font-weight: bold;
        }}
        
        .extraction-result {{
            margin: 15px 0;
            padding: 15px;
            border: 1px solid #ff00ff;
            border-radius: 5px;
            background: rgba(255, 0, 255, 0.05);
        }}
        
        .code-block {{
            background: rgba(0, 0, 0, 0.8);
            border: 1px solid #00ff00;
            padding: 15px;
            border-radius: 5px;
            font-family: 'Share Tech Mono', monospace;
            color: #00ff00;
            margin: 10px 0;
            overflow-x: auto;
        }}
        
        h1, h2, h3 {{
            font-family: 'Orbitron', monospace;
        }}
        
        h2 {{
            color: #00ffff;
            border-bottom: 2px solid #00ffff;
            padding-bottom: 5px;
        }}
        
        h3 {{
            color: #ff00ff;
        }}
        
        .footer {{
            text-align: center;
            margin-top: 50px;
            padding: 20px;
            border-top: 2px solid #00ffff;
            color: #888;
        }}
    </style>
</head>
<body>
    <div class="header">
        ⚡ DEEP ANAL ⚡<br>
        <div style="font-size: 1rem; margin-top: 10px; color: #ff00ff;">
            COMPREHENSIVE STEGANOGRAPHY ANALYSIS REPORT
        </div>
    </div>
    
    <div class="section">
        <h2>📋 ANALYSIS SUMMARY</h2>
        <div class="metric">
            <div class="metric-label">Analysis Date</div>
            <div class="metric-value">{timestamp}</div>
        </div>
        <div class="metric">
            <div class="metric-label">Target File</div>
            <div class="metric-value">{filename}</div>
        </div>
        <div class="metric">
            <div class="metric-label">File Size</div>
            <div class="metric-value">{file_size:,} bytes</div>
        </div>
        <div class="metric">
            <div class="metric-label">Entropy</div>
            <div class="metric-value">{entropy:.4f}</div>
        </div>
        
        <div class="neon-border">
            <h3>🎯 DETECTION RESULTS</h3>
            <div class="metric">
                <div class="metric-label">Steganography Likelihood</div>
                <div class="metric-value" style="color: {risk_color};">{likelihood * 100:.1f}%</div>
            </div>
            <div class="metric">
                <div class="metric-label">Risk Assessment</div>
                <div class="metric-value risk-{risk_level.lower()}">{risk_level} RISK</div>
            </div>
        </div>
    </div>"""
    
    # Add interpretation section
    html_content += f"""
    <div class="section">
        <h2>🔍 THREAT ASSESSMENT</h2>
        <div class="neon-border">"""
    
    if likelihood >= 0.7:
        html_content += """
            <div class="risk-high">
                ⚠️ HIGH PROBABILITY of steganographic content detected!<br>
                This file very likely contains hidden data and requires immediate investigation.
            </div>
            <h3>📋 Recommended Actions:</h3>
            <ul>
                <li>Immediately extract hidden content using specialized tools</li>
                <li>Analyze extracted data for sensitive information</li>
                <li>Investigate source and distribution of this file</li>
                <li>Consider security implications and containment measures</li>
                <li>Document findings for incident response team</li>
            </ul>"""
    elif likelihood >= 0.3:
        html_content += """
            <div class="risk-medium">
                ⚡ MODERATE PROBABILITY of steganographic content detected.<br>
                This file may contain hidden data and warrants further investigation.
            </div>
            <h3>📋 Recommended Actions:</h3>
            <ul>
                <li>Run additional steganography detection tools</li>
                <li>Attempt content extraction with various methods</li>
                <li>Monitor for additional suspicious files</li>
                <li>Document findings for security review</li>
                <li>Consider additional forensic analysis</li>
            </ul>"""
    else:
        html_content += """
            <div class="risk-low">
                ✅ LOW PROBABILITY of steganographic content.<br>
                This file appears to be clean of hidden data based on current analysis.
            </div>
            <h3>📋 Recommended Actions:</h3>
            <ul>
                <li>File appears clean - no immediate action required</li>
                <li>Continue standard security monitoring</li>
                <li>Retain analysis results for audit trail</li>
                <li>Periodic re-analysis may be beneficial</li>
            </ul>"""
    
    html_content += """</div></div>"""
    
    # Add file metadata section
    if metadata:
        html_content += """
    <div class="section">
        <h2>📄 FILE METADATA</h2>
        <table>
            <tr><th>Property</th><th>Value</th></tr>"""
        
        for key, value in metadata.items():
            safe_key = html.escape(str(key))
            safe_value = html.escape(str(value))
            html_content += f"<tr><td>{safe_key}</td><td>{safe_value}</td></tr>"
        
        html_content += """</table></div>"""
    
    # Add technical indicators section
    if hasattr(detection_result, 'indicators') and detection_result.indicators:
        html_content += """
    <div class="section">
        <h2>🔬 TECHNICAL INDICATORS</h2>
        <table>
            <tr><th>Indicator</th><th>Value</th><th>Weight</th><th>Impact</th></tr>"""
        
        for indicator, details in detection_result.indicators.items():
            if isinstance(details, dict):
                # Handle both 'value' and 'score' keys safely
                raw_value = details.get('value', details.get('score', 0))
                weight = details.get('weight', 'N/A')
                
                # Safe numeric conversion
                try:
                    if isinstance(raw_value, (int, float)):
                        numeric_value = float(raw_value)
                    else:
                        # Try to extract number from string like "0.85" or "85%"
                        clean_str = str(raw_value).replace('%', '').strip()
                        numeric_value = float(clean_str) if clean_str.replace('.', '').isdigit() else 0.0
                except (ValueError, AttributeError):
                    numeric_value = 0.0
                
                # Safe HTML escaping
                safe_indicator = html.escape(indicator.replace('_', ' ').title())
                safe_value = html.escape(str(raw_value))
                safe_weight = html.escape(str(weight))
                
                # Color coding based on numeric value
                impact_color = '#ff0040' if numeric_value > 0.7 else '#ff8c00' if numeric_value > 0.3 else '#00ff00'
                impact_level = 'High' if numeric_value > 0.7 else 'Medium' if numeric_value > 0.3 else 'Low'
                
                html_content += f"""<tr>
                    <td>{safe_indicator}</td>
                    <td style="color: {impact_color};">{safe_value}</td>
                    <td>{safe_weight}</td>
                    <td style="color: {impact_color};">{impact_level}</td>
                </tr>"""
        
        html_content += """</table></div>"""
    
    # Add extraction results if available
    if extracted_data and len(extracted_data) > 0:
        html_content += """
    <div class="section">
        <h2>🔓 EXTRACTION RESULTS</h2>"""
        
        for i, result in enumerate(extracted_data[:5], 1):
            success_color = '#00ff00' if result.get('success', False) else '#ff0040'
            html_content += f"""
            <div class="extraction-result">
                <h3 style="color: {success_color};">Method #{i}: {result.get('method', 'Unknown')}</h3>
                <div class="metric">
                    <div class="metric-label">Confidence</div>
                    <div class="metric-value">{result.get('confidence', 0) * 100:.1f}%</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Status</div>
                    <div class="metric-value" style="color: {success_color};">{'SUCCESS' if result.get('success', False) else 'FAILED'}</div>
                </div>"""
            
            if result.get('content'):
                content = str(result['content'])[:500]  # Limit for display
                safe_content = html.escape(content)
                truncated = '...' if len(str(result['content'])) > 500 else ''
                html_content += f"""
                <h4>Extracted Content:</h4>
                <div class="code-block">{safe_content}{truncated}</div>"""
            
            html_content += "</div>"
        
        html_content += "</div>"
    
    # Add analysis explanation if available
    if hasattr(detection_result, 'explanation'):
        safe_explanation = html.escape(str(detection_result.explanation))
        html_content += f"""
    <div class="section">
        <h2>📊 ANALYSIS EXPLANATION</h2>
        <div class="neon-border">
            {safe_explanation}
        </div>
    </div>"""
    
    # Add channel analysis if available
    if channel_analysis:
        html_content += """
    <div class="section">
        <h2>🌈 CHANNEL ANALYSIS SUMMARY</h2>
        <div class="neon-border">
            <h3>RGB Channel Statistics:</h3>
            <table>
                <tr><th>Channel</th><th>Mean</th><th>Std Dev</th><th>Entropy</th><th>Anomalies</th></tr>"""
        
        if 'red_stats' in channel_analysis:
            for channel, stats in [('Red', channel_analysis.get('red_stats', {})), 
                                   ('Green', channel_analysis.get('green_stats', {})), 
                                   ('Blue', channel_analysis.get('blue_stats', {}))]:
                anomaly_count = len(stats.get('anomalies', []))
                anomaly_color = '#ff0040' if anomaly_count > 0 else '#00ff00'
                html_content += f"""
                <tr>
                    <td>{channel}</td>
                    <td>{stats.get('mean', 'N/A')}</td>
                    <td>{stats.get('std', 'N/A')}</td>
                    <td>{stats.get('entropy', 'N/A')}</td>
                    <td style="color: {anomaly_color};">{anomaly_count} detected</td>
                </tr>"""
        
        html_content += """</table></div></div>"""
    
    # Add interactive visualizations if image_path is provided
    if image_path and os.path.exists(image_path):
        html_content += """
    <div class="section">
        <h2>📊 INTERACTIVE VISUALIZATIONS</h2>
        <p style="color: #00ffff;">The following charts are fully interactive - you can rotate 3D plots, zoom, and hover for details!</p>"""
        
        try:
            # Generate entropy visualization
            from utils.visualizations import create_entropy_plot
            entropy_fig = create_entropy_plot(entropy, lower_staging=False)
            entropy_html = entropy_fig.to_html(include_plotlyjs=False, div_id="entropy_plot", config={'responsive': True})
            html_content += f"""
        <div class="neon-border">
            <h3>🌀 3D Entropy Visualization</h3>
            <p style="color: #9400d3; font-size: 0.9rem;">Interactive 3D representation of file entropy ({entropy:.4f})</p>
            {entropy_html}
        </div>"""
        except Exception as e:
            html_content += f'<p style="color: #ff0040;">Could not generate entropy visualization: {str(e)}</p>'
        
        try:
            # Generate byte frequency plot using upgraded module
            from utils.visualizations import create_byte_frequency_plot_upgraded
            
            # Use 3D mode for the HTML report
            freq_fig = create_byte_frequency_plot_upgraded(image_path, mode='3d')
            freq_html = freq_fig.to_html(include_plotlyjs=False, div_id="frequency_plot", config={'responsive': True})
            html_content += f"""
        <div class="neon-border" style="margin-top: 20px;">
            <h3>📈 Byte Frequency Analysis (3D)</h3>
            <p style="color: #9400d3; font-size: 0.9rem;">Interactive 3D distribution of byte values throughout the file</p>
            {freq_html}
        </div>"""
        except Exception as e:
            html_content += f'<p style="color: #ff0040;">Could not generate frequency visualization: {str(e)}</p>'
        
        try:
            # Generate channel analysis if it's an image
            from utils.visualizations import create_channel_analysis_visualization
            from PIL import Image as PILImage
            
            # Check if it's a valid image
            test_img = PILImage.open(image_path)
            if test_img.mode in ['RGB', 'RGBA']:
                # Create channel plots - only do one channel to keep report size reasonable
                channel_result = create_channel_analysis_visualization(image_path, channel='red')
                if channel_result and isinstance(channel_result, dict) and 'lsb_plot' in channel_result:
                    lsb_fig = channel_result['lsb_plot']
                    channel_html = lsb_fig.to_html(include_plotlyjs=False, div_id="red_channel_lsb", config={'responsive': True})
                    html_content += f"""
        <div class="neon-border" style="margin-top: 20px;">
            <h3>🎨 Red Channel LSB Analysis</h3>
            <p style="color: #9400d3; font-size: 0.9rem;">Least significant bit distribution in red channel</p>
            {channel_html}
        </div>"""
        except Exception as e:
            pass  # Silently skip if not an image or error
        
        html_content += "</div>"
    
    # Footer
    html_content += f"""
    <div class="footer">
        <p>Generated by DEEP ANAL v1.0 - Hardcore Steganography Analysis System</p>
        <p>Report ID: {datetime.now().strftime('%Y%m%d_%H%M%S')} | Analysis Engine: Advanced Multi-Method Detection</p>
        <p>⚡ CLASSIFIED ANALYSIS COMPLETE ⚡</p>
    </div>
</body>
</html>"""
    
    return html_content

def generate_text_report(filename, detection_result, metadata, likelihood):
    """Generate human-readable text report"""
    
    report_lines = [
        "=" * 60,
        "DEEP ANAL - STEGANOGRAPHY ANALYSIS REPORT",
        "=" * 60,
        "",
        f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"File Analyzed: {filename}",
        f"Analysis Tool: DEEP ANAL v1.0",
        "",
        "DETECTION SUMMARY",
        "-" * 20,
        f"Steganography Likelihood: {likelihood * 100:.1f}%",
        f"Risk Level: {'HIGH' if likelihood >= 0.7 else 'MEDIUM' if likelihood >= 0.3 else 'LOW'}",
        "",
        "INTERPRETATION",
        "-" * 15,
    ]
    
    # Add interpretation based on likelihood
    if likelihood >= 0.7:
        report_lines.extend([
            "⚠️  HIGH PROBABILITY of hidden content detected!",
            "   This image very likely contains steganographic data.",
            "   Immediate investigation recommended.",
        ])
    elif likelihood >= 0.3:
        report_lines.extend([
            "⚡ MODERATE PROBABILITY of hidden content detected.",
            "   This image may contain steganographic data.",
            "   Further analysis recommended.",
        ])
    else:
        report_lines.extend([
            "✅ LOW PROBABILITY of steganographic content.",
            "   This image appears to be clean of hidden data.",
            "   Standard monitoring sufficient.",
        ])
    
    report_lines.extend([
        "",
        "FILE METADATA",
        "-" * 13,
    ])
    
    # Add key metadata
    for key, value in metadata.items():
        if key in ['File Size', 'Image Width', 'Image Height', 'Color Space', 'Compression']:
            report_lines.append(f"{key}: {value}")
    
    # Add technical details if available
    if hasattr(detection_result, 'indicators'):
        report_lines.extend([
            "",
            "TECHNICAL INDICATORS",
            "-" * 19,
        ])
        for indicator, details in detection_result.indicators.items():
            if isinstance(details, dict):
                score = details.get('score', 'N/A')
                weight = details.get('weight', 'N/A')
                report_lines.append(f"{indicator}: Score={score}, Weight={weight}")
    
    # Add explanation if available
    if hasattr(detection_result, 'explanation'):
        report_lines.extend([
            "",
            "ANALYSIS EXPLANATION",
            "-" * 19,
            detection_result.explanation,
        ])
    
    report_lines.extend([
        "",
        "RECOMMENDATIONS",
        "-" * 15,
    ])
    
    if likelihood >= 0.7:
        report_lines.extend([
            "1. Immediately extract hidden content using specialized tools",
            "2. Analyze extracted data for sensitive information",
            "3. Investigate source and distribution of this image",
            "4. Consider security implications and containment",
        ])
    elif likelihood >= 0.3:
        report_lines.extend([
            "1. Run additional steganography detection tools",
            "2. Attempt content extraction with various methods",
            "3. Monitor for additional suspicious images",
            "4. Document findings for security review",
        ])
    else:
        report_lines.extend([
            "1. File appears clean - no immediate action required",
            "2. Continue standard security monitoring",
            "3. Retain analysis results for audit trail",
        ])
    
    report_lines.extend([
        "",
        "=" * 60,
        "End of Report",
        "=" * 60,
    ])
    
    return "\n".join(report_lines)

# Configure Streamlit page
try:
    st.set_page_config(
        page_title="DEEP ANAL: Hardcore Steganography Analysis",
        page_icon="⚡",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
except Exception as e:
    st.error(f"Page configuration error: {e}")
    st.stop()

# Load cyberpunk CSS theme
load_css()

# Cache busting banner - prominently displayed at top
st.markdown(f"""
<div style="background: linear-gradient(135deg, #ff0040 0%, #ff00ff 100%); 
     padding: 15px; border-radius: 8px; margin-bottom: 20px; border: 2px solid #00ffff;">
    <h3 style="color: #fff; margin: 0; text-align: center; font-family: 'Share Tech Mono', monospace;">
        🚀 DEEP ANAL v{APP_VERSION} - UNIVERSAL FILE SUPPORT ACTIVE
    </h3>
    <p style="color: #fff; margin: 10px 0 0 0; text-align: center; font-size: 0.9rem;">
        📢 <strong>Seeing file upload restrictions?</strong> Press 
        <code style="background: #000; padding: 2px 8px; border-radius: 3px;">Ctrl+Shift+R</code> (Windows/Linux) or 
        <code style="background: #000; padding: 2px 8px; border-radius: 3px;">Cmd+Shift+R</code> (Mac) to load the new version!
    </p>
</div>
""", unsafe_allow_html=True)

# Add clear cache button
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    if st.button("🔄 FORCE RELOAD - Clear All Caches", use_container_width=True, type="primary"):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.rerun()

# Cyberpunk header
st.markdown("""
<div class="main-header">
    <h1>⚡ DEEP ANAL ⚡</h1>
    <p style="color: #ff00ff; font-family: 'Share Tech Mono', monospace; text-align: center; font-size: 1.2rem; margin-top: -1rem;">
        HARDCORE STEGANOGRAPHY ANALYSIS SYSTEM
    </p>
    <p style="color: #00ffff; font-family: 'Share Tech Mono', monospace; text-align: center; font-size: 0.9rem;">
        DETECTING HIDDEN DATA • EXTRACTING SECRETS • ANALYZING THREATS
    </p>
</div>
""", unsafe_allow_html=True)

# Cyberpunk upload mode selection
st.markdown("<div class='terminal-panel'>", unsafe_allow_html=True)
st.markdown("**>>> SELECT OPERATION MODE:**")
upload_mode = st.radio(
    "Choose your mission:",
    ["⚡ SINGLE TARGET ANALYSIS", "🔥 MASS SURVEILLANCE SCAN"],
    horizontal=True,
    label_visibility="collapsed"
)
st.markdown("</div>", unsafe_allow_html=True)

uploaded_file = None
if upload_mode == "⚡ SINGLE TARGET ANALYSIS":
    # Cyberpunk single file upload interface
    create_terminal_panel(
        "FILE ACQUISITION MODULE",
        """
        <p style='color: #00ff00; font-family: "Share Tech Mono", monospace;'>
        >>> INITIALIZING DEEP SCAN PROTOCOL...<br>
        >>> UNIVERSAL FILE SUPPORT: ALL FILE TYPES ACCEPTED<br>
        >>> IMAGES | VIDEOS | DOCUMENTS | ARCHIVES | EXECUTABLES | RAW DATA<br>
        >>> DRAG TARGET FILE TO INITIATE ANALYSIS SEQUENCE
        </p>
        """,
        "online"
    )
    
    uploaded_file = st.file_uploader(
        ">>> DEPLOY TARGET FILE FOR STEGANOGRAPHIC INTERROGATION",
        help="CLASSIFIED ANALYSIS PROTOCOLS ACTIVE - ALL FILE TYPES ACCEPTED",
        label_visibility="collapsed"
    )
else:
    # Cyberpunk batch processing interface
    create_terminal_panel(
        "MASS SURVEILLANCE PROTOCOL",
        """
        <p style='color: #ff00ff; font-family: "Share Tech Mono", monospace;'>
        >>> INITIALIZING MASS DATA ACQUISITION...<br>
        >>> BATCH PROCESSING MODE: ACTIVE<br>
        >>> THREAT DETECTION ALGORITHMS: LOADED<br>
        >>> READY FOR MULTI-TARGET ANALYSIS
        </p>
        """,
        "online"
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**⚡ PROTOCOL ALPHA: MULTI-TARGET UPLOAD**")
        uploaded_files = st.file_uploader(
            ">>> DEPLOY MULTIPLE TARGETS FOR BATCH INTERROGATION",
            accept_multiple_files=True,
            help="CLASSIFIED: MASS SURVEILLANCE ANALYSIS - ALL FILE TYPES ACCEPTED",
            label_visibility="collapsed"
        )
    
    with col2:
        st.markdown("**🔥 PROTOCOL BETA: ARCHIVE DECOMPRESSION**")
        uploaded_zip = st.file_uploader(
            ">>> DEPLOY COMPRESSED ARCHIVE FOR EXTRACTION",
            help="CLASSIFIED: COMPRESSED/ARCHIVE ANALYSIS - ALL FORMATS ACCEPTED",
            label_visibility="collapsed"
        )
    
    # Process ZIP file if uploaded
    if uploaded_zip:
        import zipfile
        import io
        
        try:
            with zipfile.ZipFile(io.BytesIO(uploaded_zip.getvalue())) as zip_ref:
                # Extract ALL files from ZIP (universal file support)
                extracted_files = []
                
                for file_info in zip_ref.filelist:
                    if not file_info.is_dir():
                        try:
                            file_data = zip_ref.read(file_info.filename)
                            # Create a file-like object that mimics uploaded_file
                            class ZipExtractedFile:
                                def __init__(self, name, data):
                                    self.name = name
                                    self.data = data
                                
                                def read(self):
                                    return self.data
                                
                                def getvalue(self):
                                    return self.data
                            
                            extracted_files.append(ZipExtractedFile(file_info.filename, file_data))
                        except Exception as e:
                            st.warning(f"Failed to extract {file_info.filename}: {str(e)}")
                
                if extracted_files:
                    st.success(f"📦 Extracted {len(extracted_files)} files from ZIP archive (ALL file types supported)")
                    uploaded_files = extracted_files  # Use extracted files for batch processing
                else:
                    st.error("No files found in ZIP archive")
                    uploaded_files = None
                    
        except zipfile.BadZipFile:
            st.error("Invalid ZIP file. Please upload a valid ZIP archive.")
            uploaded_files = None
        except Exception as e:
            st.error(f"Error processing ZIP file: {str(e)}")
            uploaded_files = None
    
    if uploaded_files:
        st.info(f"📁 Ready to scan {len(uploaded_files)} files")
        
        if st.button("🚀 Start Batch Scan", type="primary"):
            # Process files in batch
            batch_results = []
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, uploaded_file in enumerate(uploaded_files):
                status_text.text(f"Scanning {uploaded_file.name}... ({i+1}/{len(uploaded_files)})")
                
                try:
                    # Create temporary file
                    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
                        tmp_file.write(uploaded_file.read())
                        temp_path = tmp_file.name
                    
                    # Check if it's a video file and process accordingly
                    if is_video_file(temp_path):
                        # For videos, extract frames and analyze the first few
                        frames, frame_msg = extract_video_frames(temp_path, max_frames=3)
                        if frames:
                            # Analyze the first frame
                            frame_path = save_video_frame_for_analysis(frames[0])
                            if frame_path:
                                detection_result = analyze_image_for_steganography(frame_path)
                                metadata = analyze_video_metadata(temp_path)
                                os.unlink(frame_path)  # Cleanup frame
                            else:
                                detection_result = None
                                metadata = analyze_video_metadata(temp_path)
                        else:
                            detection_result = None
                            metadata = analyze_video_metadata(temp_path)
                    else:
                        # Regular image analysis
                        detection_result = analyze_image_for_steganography(temp_path)
                        metadata = get_file_metadata(temp_path)
                    
                    if detection_result and hasattr(detection_result, 'likelihood'):
                        likelihood = detection_result.likelihood
                    else:
                        likelihood = 0.0
                    
                    # Store result
                    batch_results.append({
                        "Filename": uploaded_file.name,
                        "Likelihood": likelihood,
                        "Percentage": f"{likelihood * 100:.1f}%",
                        "Risk Level": "🔴 High" if likelihood >= 0.7 else "🟡 Medium" if likelihood >= 0.3 else "🟢 Low",
                        "File Size": f"{len(uploaded_file.getvalue()) / 1024:.1f} KB"
                    })
                    
                    # Cleanup
                    try:
                        os.unlink(temp_path)
                    except:
                        pass
                        
                except Exception as e:
                    batch_results.append({
                        "Filename": uploaded_file.name,
                        "Likelihood": 0.0,
                        "Percentage": "Error",
                        "Risk Level": "❌ Failed",
                        "File Size": f"{len(uploaded_file.getvalue()) / 1024:.1f} KB"
                    })
                
                # Update progress
                progress_bar.progress((i + 1) / len(uploaded_files))
            
            # Clear status
            status_text.empty()
            progress_bar.empty()
            
            # Sort by likelihood (high to low)
            batch_results.sort(key=lambda x: x["Likelihood"], reverse=True)
            
            # Display results
            st.success(f"✅ Batch scan complete! Processed {len(uploaded_files)} files")
            
            # Summary statistics
            col1, col2, col3, col4 = st.columns(4)
            
            high_risk = len([r for r in batch_results if r["Likelihood"] >= 0.7])
            medium_risk = len([r for r in batch_results if 0.3 <= r["Likelihood"] < 0.7])
            low_risk = len([r for r in batch_results if r["Likelihood"] < 0.3])
            
            with col1:
                st.metric("🔴 High Risk", high_risk)
            with col2:
                st.metric("🟡 Medium Risk", medium_risk)
            with col3:
                st.metric("🟢 Low Risk", low_risk)
            with col4:
                avg_likelihood = sum([r["Likelihood"] for r in batch_results]) / len(batch_results) if batch_results else 0
                st.metric("📊 Avg Likelihood", f"{avg_likelihood * 100:.1f}%")
            
            st.markdown("---")
            st.subheader("📋 Scan Results (Sorted by Likelihood)")
            
            # Create DataFrame and display
            import pandas as pd
            df = pd.DataFrame([{k: v for k, v in result.items() if k != "Likelihood"} for result in batch_results])
            
            # Style the dataframe
            def highlight_risk(row):
                if "🔴 High" in str(row["Risk Level"]):
                    return ['background-color: rgba(255, 0, 0, 0.1)'] * len(row)
                elif "🟡 Medium" in str(row["Risk Level"]):
                    return ['background-color: rgba(255, 255, 0, 0.1)'] * len(row)
                else:
                    return ['background-color: rgba(0, 255, 0, 0.1)'] * len(row)
            
            styled_df = df.style.apply(highlight_risk, axis=1)
            st.dataframe(styled_df, use_container_width=True, height=400)
            
            # Download batch results
            st.markdown("---")
            col1, col2 = st.columns([1, 1])
            
            with col1:
                # CSV download
                csv = df.to_csv(index=False)
                st.download_button(
                    label="📄 Download Results (CSV)",
                    data=csv,
                    file_name=f"batch_steganography_scan_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
            
            with col2:
                # JSON download with full data
                json_data = {
                    "scan_metadata": {
                        "scan_date": datetime.now().isoformat(),
                        "total_files": len(uploaded_files),
                        "high_risk_count": high_risk,
                        "medium_risk_count": medium_risk,
                        "low_risk_count": low_risk,
                        "average_likelihood": avg_likelihood
                    },
                    "results": batch_results
                }
                json_str = json.dumps(json_data, indent=2)
                st.download_button(
                    label="📋 Download Full Report (JSON)",
                    data=json_str,
                    file_name=f"batch_steganography_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
            
            # High-risk file recommendations
            high_risk_files = [r for r in batch_results if r["Likelihood"] >= 0.7]
            if high_risk_files:
                st.markdown("---")
                st.subheader("🚨 High-Risk Files Detected")
                st.warning(f"Found {len(high_risk_files)} files with high steganography likelihood. Immediate investigation recommended:")
                
                for file_result in high_risk_files[:5]:  # Show top 5
                    st.write(f"• **{file_result['Filename']}** - {file_result['Percentage']} likelihood")
                
                st.info("💡 Tip: Use Single File Analysis mode to perform detailed extraction on these high-risk files.")
        
        # Exit early for batch mode
        st.stop()

# Continue with single file analysis if a file was uploaded
if upload_mode == "⚡ SINGLE TARGET ANALYSIS" and uploaded_file:
    # Create temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        temp_path = tmp_file.name
    
    # File identification using magic numbers
    st.markdown("---")
    st.subheader("🔍 FILE IDENTIFICATION")
    
    file_info = identify_file_type(temp_path)
    safety_check = is_safe_to_analyze(temp_path)
    signature = get_file_signature(temp_path)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if "error" not in file_info:
            st.metric("📁 File Type", file_info.get('category', 'unknown').upper())
            st.write(f"**MIME:** `{file_info.get('mime_type', 'unknown')}`")
        else:
            st.error(f"File type unknown: {file_info['error']}")
    
    with col2:
        file_size_mb = len(uploaded_file.getvalue()) / (1024 * 1024)
        st.metric("📊 File Size", f"{file_size_mb:.2f} MB")
        if "error" not in file_info:
            st.write(f"**Binary:** {'Yes' if file_info.get('is_binary') else 'No'}")
    
    with col3:
        if safety_check.get('is_safe'):
            st.metric("✅ Safety", "SAFE")
        else:
            st.metric("⚠️ Safety", "WARNINGS")
            for warning in safety_check.get('warnings', []):
                st.warning(warning)
    
    # Detailed file info expander
    with st.expander("🔬 Detailed File Analysis"):
        st.write("**File Description:**")
        if "error" not in file_info:
            st.code(file_info.get('description', 'Unknown'), language='text')
        
        st.write("**File Signature (Magic Bytes):**")
        if "error" not in signature:
            col_sig1, col_sig2 = st.columns(2)
            with col_sig1:
                st.code(f"HEX: {signature['hex']}", language='text')
            with col_sig2:
                st.code(f"ASCII: {signature['ascii']}", language='text')
        
        if "error" not in file_info:
            st.write("**File Characteristics:**")
            characteristics = []
            if file_info.get('is_binary'):
                characteristics.append("🔹 Binary file")
            if file_info.get('is_executable'):
                characteristics.append("⚠️ Executable")
            if file_info.get('is_compressed'):
                characteristics.append("📦 Compressed/Archive")
            if file_info.get('is_encrypted'):
                characteristics.append("🔐 Encrypted")
            
            for char in characteristics:
                st.write(char)
    
    st.markdown("---")
    
    # Check if it's a video file
    is_video = is_video_file(temp_path)
    
    if is_video:
        st.info("🎬 Video file detected. Extracting frames for steganography analysis...")
        frames, frame_msg = extract_video_frames(temp_path, max_frames=5)
        st.write(f"📱 {frame_msg}")
        
        if frames:
            st.success(f"✅ Successfully extracted {len(frames)} frames for analysis")
        else:
            st.error("❌ Could not extract frames from video")
    else:
        frames = None

    try:
        # Run analysis
        file_size = os.path.getsize(temp_path)
        file_type = Path(uploaded_file.name).suffix.lower()[1:]
        entropy_value = calculate_entropy(temp_path)
        metadata = get_file_metadata(temp_path)
        # Check if format is supported
        base_formats = ['png', 'jpg', 'jpeg', 'tiff', 'tif', 'bmp', 'webp', 'gif']
        heif_formats = ['heic', 'heif'] if HEIF_AVAILABLE else []
        supported_formats = base_formats + heif_formats
        is_image = file_type in supported_formats
        
        if is_image:
            # Steganography detection
            try:
                with st.spinner("Running steganography analysis..."):
                    detection_result = analyze_image_for_steganography(temp_path)
                likelihood = detection_result.likelihood
                likelihood_percentage = f"{likelihood*100:.1f}%"
                color = "#00ff00" if likelihood < 0.3 else "#ffff00" if likelihood < 0.6 else "#ff0000"
            except Exception as e:
                st.error(f"Analysis error: {str(e)}")
                likelihood = 0
                likelihood_percentage = "0.0%"
                color = "#00ff00"
                detection_result = None
            
            # Save to database
            if DB_AVAILABLE:
                try:
                    metadata_json = json.dumps(metadata)
                    save_analysis(uploaded_file.name, file_size, file_type, entropy_value, metadata_json)
                except Exception as e:
                    st.warning(f"Database save failed: {str(e)}")
        
            # Results display
            st.success(f"✓ Analysis Complete: {uploaded_file.name}")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("File Size", f"{file_size} bytes")
            with col2:
                st.metric("Type", file_type.upper())
            with col3:
                st.metric("Entropy", f"{entropy_value:.4f}")
            with col4:
                st.metric("Stego Detection", likelihood_percentage, delta=None)
            
            # Main analysis tabs
            tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Visualizations", "🔍 Detection Details", "📄 Metadata", "🔬 Advanced", "🌈 Channel Analysis"])
            
            with tab1:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Entropy Analysis")
                    entropy_plot = create_entropy_plot(entropy_value)
                    st.plotly_chart(entropy_plot, use_container_width=True)
                    
                with col2:
                    st.subheader("Byte Frequency")
                    bytes_values, frequencies = get_byte_frequency(temp_path)
                    freq_plot = create_byte_frequency_plot(bytes_values, frequencies)
                    st.plotly_chart(freq_plot, use_container_width=True)
                
                st.markdown("---")
                st.subheader("🆕 Advanced Visualization Modules")
                
                # New Byte Frequency Module (Upgraded with 2D/3D toggle)
                with st.expander("📊 Byte Frequency Analysis [UPDATED]", expanded=False):
                    st.write("**Toggle between 2D heatmap and 3D bar graph visualization**")
                    viz_mode = st.radio("Visualization Mode:", ['3D Bar Graph', '2D Heatmap'], horizontal=True)
                    mode = '3d' if viz_mode == '3D Bar Graph' else '2d'
                    
                    try:
                        freq_upgraded = create_byte_frequency_plot_upgraded(temp_path, mode=mode)
                        st.plotly_chart(freq_upgraded, use_container_width=True)
                    except Exception as e:
                        st.error(f"Error generating byte frequency plot: {str(e)}")
                
                # Bitplane Visualizer
                with st.expander("🔲 Bitplane Analysis (24 Layers)", expanded=False):
                    st.write("**Visualize individual bit layers across RGB channels**")
                    group_mode = st.selectbox("Bitplane Group:", ['LSB (Bits 0-3)', 'MSB (Bits 4-7)', 'All 24 Bitplanes'])
                    mode_map = {'LSB (Bits 0-3)': 'lsb', 'MSB (Bits 4-7)': 'msb', 'All 24 Bitplanes': 'all'}
                    
                    try:
                        bitplane_fig = create_bitplane_visualizer(temp_path, group_mode=mode_map[group_mode])
                        st.plotly_chart(bitplane_fig, use_container_width=True)
                        st.caption("White pixels = bit value 1, Black pixels = bit value 0")
                    except Exception as e:
                        st.error(f"Error generating bitplane visualization: {str(e)}")
                
                # RGB 3D Scatter Plot
                with st.expander("🌈 RGB Color Space Distribution", expanded=False):
                    st.write("**Map each pixel's RGB values into 3D color space**")
                    col_a, col_b = st.columns(2)
                    with col_a:
                        sample_size = st.slider("Sample Size:", 1000, 10000, 5000, step=1000)
                    with col_b:
                        enable_density = st.checkbox("Enable Density Smoothing", value=True)
                    
                    try:
                        rgb_scatter = create_rgb_3d_scatter(temp_path, sample_size=sample_size, enable_density=enable_density)
                        st.plotly_chart(rgb_scatter, use_container_width=True)
                    except Exception as e:
                        st.error(f"Error generating RGB scatter plot: {str(e)}")
                
                # Entropy Terrain Map
                with st.expander("🗺️ Entropy Terrain Map", expanded=False):
                    st.write("**Block-based Shannon entropy visualization as 3D heightmap**")
                    block_size = st.select_slider("Block Size:", options=[8, 16, 32, 64], value=16)
                    
                    try:
                        entropy_terrain = create_entropy_terrain_map(temp_path, block_size=block_size)
                        st.plotly_chart(entropy_terrain, use_container_width=True)
                        st.caption("Higher peaks indicate higher entropy (more randomness) in that region")
                    except Exception as e:
                        st.error(f"Error generating entropy terrain map: {str(e)}")
                
                # Segment Structure Mapper
                with st.expander("🧩 File Structure Map", expanded=False):
                    st.write("**Parse and visualize file format structure (chunks/segments)**")
                    
                    try:
                        structure_map = create_segment_structure_mapper(temp_path)
                        st.plotly_chart(structure_map, use_container_width=True)
                        st.caption("Shows internal file structure: PNG chunks, JPEG markers, or generic blocks")
                    except Exception as e:
                        st.error(f"Error generating structure map: {str(e)}")
            
            with tab2:
                st.subheader("Steganography Detection Results")
                
                if detection_result and hasattr(detection_result, 'indicators'):
                    # Simple table display
                    import pandas as pd
                    
                    indicator_data = []
                    for name, details in detection_result.indicators.items():
                        indicator_data.append({
                            "Test": name.replace('_', ' ').title(),
                            "Score": f"{details['value']:.3f}",
                            "Weight": f"{details['weight']:.1f}"
                        })
                    
                    df = pd.DataFrame(indicator_data)
                    st.dataframe(df, use_container_width=True)
                    
                    st.write("**Overall Likelihood:**", likelihood_percentage)
                    if hasattr(detection_result, 'explanation'):
                        st.write("**Analysis:**", detection_result.explanation)
                        
                    # Add extraction functionality if likelihood is high enough
                    if likelihood >= 0.4:
                        st.markdown("---")
                        st.subheader("🔓 Hidden Content Extraction")
                        st.write("The analysis suggests possible hidden data. Try these extraction methods:")
                        
                        col1, col2, col3 = st.columns(3)
                        
                        # Quick extraction buttons
                        with col1:
                            if st.button("🎯 Auto Extract", help="Try multiple methods automatically"):
                                with st.spinner("Attempting automatic extraction..."):
                                    try:
                                        results = brute_force_decode(temp_path)
                                        successful_results = [r for r in results if r.success and r.confidence > 0.3]
                                        
                                        # Store extraction results in session state for HTML report
                                        if successful_results:
                                            extraction_data = []
                                            for result in successful_results:
                                                extraction_data.append({
                                                    'method': result.method,
                                                    'success': result.success,
                                                    'confidence': result.confidence,
                                                    'content': result.data.decode('utf-8', errors='ignore')[:1000] if result.data else None,
                                                    'data_size': len(result.data) if result.data else 0
                                                })
                                            st.session_state.extraction_results = extraction_data
                                        
                                        if successful_results:
                                            st.success(f"✅ Found {len(successful_results)} potential hidden content(s)!")
                                            
                                            # Add CSV download for multiple results
                                            if len(successful_results) > 1:
                                                csv_data, csv_filename = create_bulk_extraction_csv(successful_results, "auto_extraction_results")
                                                st.download_button(
                                                    label="📊 Download All Results (CSV)",
                                                    data=csv_data,
                                                    file_name=csv_filename,
                                                    mime="text/csv",
                                                    help="Download summary of all extraction results",
                                                    use_container_width=False
                                                )
                                            
                                            for i, result in enumerate(successful_results[:3]):  # Show top 3
                                                st.write(f"**Method {i+1}: {result.method}** (Confidence: {result.confidence:.2f})")
                                                
                                                # Display extracted data
                                                if result.data:
                                                    try:
                                                        # Try to decode as text first
                                                        text_data = result.data.decode('utf-8', errors='ignore')
                                                        if len(text_data.strip()) > 0 and all(ord(c) < 127 for c in text_data[:100]):
                                                            st.text_area(f"Extracted Text {i+1}:", text_data[:1000], height=100)
                                                            st.markdown("**💾 Download Options:**")
                                                            create_extraction_download_buttons(text_data, f"{result.method}_text_{i+1}", is_text=True)
                                                        else:
                                                            st.write(f"**Binary data found:** {len(result.data)} bytes")
                                                            # Show hex preview
                                                            hex_preview = ' '.join(f'{b:02x}' for b in result.data[:32])
                                                            st.code(f"Hex preview: {hex_preview}{'...' if len(result.data) > 32 else ''}")
                                                            
                                                            st.markdown("**💾 Download Options:**")
                                                            create_extraction_download_buttons(result.data, f"{result.method}_binary_{i+1}", is_text=False)
                                                    except:
                                                        st.write(f"**Binary data found:** {len(result.data)} bytes")
                                                        st.markdown("**💾 Download Options:**")
                                                        create_extraction_download_buttons(result.data, f"{result.method}_binary_{i+1}", is_text=False)
                                        else:
                                            st.warning("No clear hidden content found with automatic methods")
                                    except Exception as e:
                                        st.error(f"Extraction failed: {str(e)}")
                        
                        with col2:
                            if st.button("🔍 LSB Extract", help="Extract using LSB steganography"):
                                with st.spinner("Extracting LSB data..."):
                                    try:
                                        # Try different LSB configurations
                                        best_result = None
                                        best_confidence = 0
                                        
                                        for channel in range(3):  # RGB channels
                                            for bit_plane in [0, 1]:  # LSB and second bit
                                                result = decode_lsb(temp_path, bit_plane, channel)
                                                if result.confidence > best_confidence:
                                                    best_result = result
                                                    best_confidence = result.confidence
                                        
                                        if best_result and best_result.confidence > 0.3:
                                            st.success(f"✅ LSB extraction successful! (Confidence: {best_result.confidence:.2f})")
                                            st.write(f"**Method:** {best_result.method}")
                                            
                                            if best_result.data:
                                                try:
                                                    text_data = best_result.data.decode('utf-8', errors='ignore')
                                                    if len(text_data.strip()) > 0:
                                                        st.text_area("Extracted LSB Text:", text_data[:1000], height=100)
                                                        st.markdown("**💾 Download Options:**")
                                                        create_extraction_download_buttons(text_data, "LSB_extraction", is_text=True)
                                                    else:
                                                        st.write(f"**Binary LSB data:** {len(best_result.data)} bytes")
                                                        st.markdown("**💾 Download Options:**")
                                                        create_extraction_download_buttons(best_result.data, "LSB_binary", is_text=False)
                                                except:
                                                    st.write(f"**Binary LSB data:** {len(best_result.data)} bytes")
                                                    # Save binary data to file for external analysis
                                                    saved_file = save_extracted_binary(best_result.data, "LSB", 2)
                                                    if saved_file:
                                                        st.success(f"💾 Binary data saved to `{saved_file}` for external analysis")
                                                        st.code(f"Analyze with: file {saved_file} && binwalk {saved_file} && strings {saved_file}")
                                        else:
                                            st.warning("No clear LSB hidden content found")
                                    except Exception as e:
                                        st.error(f"LSB extraction failed: {str(e)}")
                        
                        with col3:
                            if st.button("🛠️ Steghide Extract", help="Extract using Steghide tool"):
                                with st.spinner("Attempting Steghide extraction..."):
                                    try:
                                        # Try with no password first
                                        result = try_steghide_extract(temp_path, "")
                                        
                                        if result.success:
                                            st.success("✅ Steghide extraction successful!")
                                            if result.data:
                                                try:
                                                    text_data = result.data.decode('utf-8', errors='ignore')
                                                    if len(text_data.strip()) > 0:
                                                        st.text_area("Extracted Steghide Text:", text_data, height=100)
                                                        st.markdown("**💾 Download Options:**")
                                                        create_extraction_download_buttons(text_data, "Steghide_extraction", is_text=True)
                                                    else:
                                                        st.write(f"**Binary data extracted:** {len(result.data)} bytes")
                                                        st.markdown("**💾 Download Options:**")
                                                        create_extraction_download_buttons(result.data, "Steghide_binary", is_text=False)
                                                        # Offer download
                                                        st.download_button(
                                                            "Download extracted file",
                                                            result.data,
                                                            file_name="extracted_content.bin",
                                                            mime="application/octet-stream"
                                                        )
                                                except:
                                                    st.write(f"**Binary data extracted:** {len(result.data)} bytes")
                                        else:
                                            st.warning("No Steghide hidden content found (no password)")
                                            
                                            # Offer password input
                                            password = st.text_input("Try with password:", type="password", key="steghide_pass")
                                            if password and st.button("Extract with password", key="steghide_pass_btn"):
                                                pass_result = try_steghide_extract(temp_path, password)
                                                if pass_result.success:
                                                    st.success("✅ Steghide extraction with password successful!")
                                                    if pass_result.data:
                                                        try:
                                                            text_data = pass_result.data.decode('utf-8', errors='ignore')
                                                            st.text_area("Extracted Text:", text_data, height=100)
                                                            st.markdown("**💾 Download Options:**")
                                                            create_extraction_download_buttons(text_data, "Steghide_password_extraction", is_text=True)
                                                        except:
                                                            st.write(f"**Binary data extracted:** {len(pass_result.data)} bytes")
                                                            st.markdown("**💾 Download Options:**")
                                                            create_extraction_download_buttons(pass_result.data, "Steghide_password_binary", is_text=False)
                                                else:
                                                    st.error("Password extraction failed")
                                    except Exception as e:
                                        st.error(f"Steghide extraction failed: {str(e)}")
                        
                        # Additional extraction methods  
                        st.markdown("---")
                        col4, col5 = st.columns(2)
                        
                        with col4:
                            if st.button("📝 OCR Extract", help="Extract text using Optical Character Recognition"):
                                with st.spinner("Running OCR analysis..."):
                                    try:
                                        ocr_result = extract_text_with_ocr(temp_path)
                                        
                                        if "error" not in ocr_result:
                                            st.success("✅ **OCR extraction completed!**")
                                            
                                            # Display OCR statistics
                                            st.write(f"**Words found:** {ocr_result['word_count']}")
                                            st.write(f"**Average confidence:** {ocr_result['average_confidence']:.1f}%")
                                            
                                            if ocr_result['raw_text']:
                                                st.text_area("Raw OCR Text:", ocr_result['raw_text'][:1000], height=150)
                                                
                                                # Analyze text for steganographic patterns
                                                text_analysis = analyze_text_for_steganography(ocr_result['raw_text'])
                                                
                                                if text_analysis['likelihood'] > 0.3:
                                                    st.warning(f"🔍 **Steganographic patterns detected!** (Likelihood: {text_analysis['likelihood']:.2f})")
                                                    st.write("**Indicators found:**")
                                                    for indicator in text_analysis['indicators']:
                                                        st.write(f"• {indicator}")
                                                
                                                # Display PGP analysis if detected
                                                if 'pgp_analysis' in text_analysis:
                                                    pgp = text_analysis['pgp_analysis']
                                                    st.markdown("---")
                                                    st.markdown("### 🔐 PGP/GPG Cryptographic Content Detected")
                                                    
                                                    # Risk level badge
                                                    risk_level = pgp.get('risk_level', 'low')
                                                    risk_colors = {
                                                        'critical': '🔴',
                                                        'high': '🟠',
                                                        'medium': '🟡',
                                                        'low': '🟢'
                                                    }
                                                    st.write(f"**Risk Level:** {risk_colors.get(risk_level, '⚪')} {risk_level.upper()}")
                                                    
                                                    # Summary
                                                    st.info(pgp.get('summary', ''))
                                                    
                                                    # Indicators
                                                    if pgp.get('indicators'):
                                                        st.write("**Security Indicators:**")
                                                        for ind in pgp['indicators']:
                                                            st.write(f"• {ind}")
                                                    
                                                    # Blocks detail
                                                    if pgp.get('blocks'):
                                                        with st.expander(f"📋 Detected PGP Blocks ({len(pgp['blocks'])} total)"):
                                                            for i, block in enumerate(pgp['blocks'], 1):
                                                                st.write(f"**Block #{i}: {block['type']}**")
                                                                if block.get('version'):
                                                                    st.write(f"  - Version: {block['version']}")
                                                                if block.get('key_id'):
                                                                    st.write(f"  - Key ID: `{block['key_id']}`")
                                                                if block.get('size_bytes'):
                                                                    st.write(f"  - Size: {block['size_bytes']} bytes")
                                                                if block.get('checksum'):
                                                                    st.write(f"  - Checksum: `{block['checksum']}`")
                                                                
                                                                # Show content preview
                                                                if block.get('content_preview'):
                                                                    st.code(block['content_preview'], language='text')
                                                                
                                                                if i < len(pgp['blocks']):
                                                                    st.markdown("---")
                                                    
                                                    # Recommendations
                                                    if pgp.get('recommendations'):
                                                        with st.expander("🔍 Investigation Recommendations"):
                                                            for rec in pgp['recommendations']:
                                                                st.write(f"• {rec}")
                                                
                                                # Save text as binary for external analysis
                                                saved_file = save_extracted_binary(ocr_result['raw_text'].encode('utf-8'), "ocr_text", 10)
                                                if saved_file:
                                                    st.info(f"💾 OCR text saved as: `{saved_file}`")
                                            else:
                                                st.info("No text detected in the image")
                                        else:
                                            st.error(f"OCR failed: {ocr_result['error']}")
                                            
                                    except Exception as e:
                                        st.error(f"OCR extraction failed: {str(e)}")
                        
                        with col5:
                            if st.button("⚡ XOR Analysis", help="Try XOR decoding on extracted data"):
                                with st.spinner("Running XOR analysis..."):
                                    try:
                                        xor_results = extract_with_xor_analysis(temp_path)
                                        
                                        if xor_results:
                                            successful_results = [r for r in xor_results if r.success]
                                            
                                            if successful_results:
                                                st.success(f"✅ **XOR analysis found {len(successful_results)} potential results!**")
                                                
                                                # Show top 3 results
                                                for i, result in enumerate(successful_results[:3]):
                                                    st.write(f"**Result #{i+1}: {result.method}**")
                                                    st.write(f"Confidence: {result.confidence:.2f}")
                                                    
                                                    if result.data:
                                                        st.markdown("**💾 Download Options:**")
                                                        create_extraction_download_buttons(result.data, f"XOR_result_{i+1}", is_text=False)
                                                        
                                                        try:
                                                            # Try to display as text
                                                            text_data = result.data.decode('utf-8', errors='ignore')
                                                            if text_data.strip() and len(text_data) < 500:
                                                                st.text_area(f"Decoded Text #{i+1}:", text_data, height=80, key=f"xor_text_{i}")
                                                            else:
                                                                st.write(f"Binary data ({len(result.data)} bytes) - Hex: {result.data[:20].hex()}...")
                                                        except:
                                                            st.write(f"Binary data ({len(result.data)} bytes) - Hex: {result.data[:20].hex()}...")
                                                    
                                                    if i < 2:  # Don't add separator after last item
                                                        st.markdown("---")
                                            else:
                                                st.info("No meaningful XOR decoding results found")
                                        else:
                                            st.warning("XOR analysis completed but no results found")
                                            
                                    except Exception as e:
                                        st.error(f"XOR analysis failed: {str(e)}")
                        
                        # Additional extraction methods
                        if st.expander("🔬 Advanced Extraction Methods"):
                            st.write("**Metadata Extraction:**")
                            if st.button("Extract from Metadata", key="metadata_extract"):
                                with st.spinner("Checking metadata for hidden data..."):
                                    try:
                                        result = extract_metadata_hidden_data(temp_path)
                                        if result.success:
                                            st.success("✅ Hidden data found in metadata!")
                                            if result.data:
                                                try:
                                                    text_data = result.data.decode('utf-8', errors='ignore')
                                                    st.text_area("Metadata Hidden Text:", text_data, height=100)
                                                    st.markdown("**💾 Download Options:**")
                                                    create_extraction_download_buttons(text_data, "Metadata_extraction", is_text=True)
                                                except:
                                                    st.write(f"**Binary metadata:** {len(result.data)} bytes")
                                                    st.markdown("**💾 Download Options:**")
                                                    create_extraction_download_buttons(result.data, "Metadata_binary", is_text=False)
                                        else:
                                            st.info("No hidden data found in metadata")
                                    except Exception as e:
                                        st.error(f"Metadata extraction failed: {str(e)}")
                            
                            st.write("**Multi-bit LSB Extraction:**")
                            bits_to_extract = st.selectbox("Number of bits to extract:", [1, 2, 3, 4], index=1)
                            channel_to_extract = st.selectbox("Color channel:", ["Red (0)", "Green (1)", "Blue (2)"], index=0)
                            channel_num = int(channel_to_extract.split("(")[1].split(")")[0])
                            
                            if st.button("Extract Multi-bit LSB", key="multibit_extract"):
                                with st.spinner(f"Extracting {bits_to_extract}-bit LSB from {channel_to_extract.split('(')[0].strip()} channel..."):
                                    try:
                                        result = decode_multi_bit_lsb(temp_path, bits_to_extract, channel_num)
                                        if result.success and result.confidence > 0.3:
                                            st.success(f"✅ Multi-bit LSB extraction successful! (Confidence: {result.confidence:.2f})")
                                            if result.data:
                                                try:
                                                    text_data = result.data.decode('utf-8', errors='ignore')
                                                    if len(text_data.strip()) > 0:
                                                        st.text_area("Extracted Multi-bit Text:", text_data[:1000], height=100)
                                                        st.markdown("**💾 Download Options:**")
                                                        create_extraction_download_buttons(text_data, "Multi_bit_LSB_extraction", is_text=True)
                                                    else:
                                                        st.write(f"**Binary data:** {len(result.data)} bytes")
                                                        st.markdown("**💾 Download Options:**")
                                                        create_extraction_download_buttons(result.data, "Multi_bit_LSB_binary", is_text=False)
                                                except:
                                                    st.write(f"**Binary data:** {len(result.data)} bytes")
                                                    # Save binary data to file for external analysis
                                                    saved_file = save_extracted_binary(result.data, "Multi_LSB", 6)
                                                    if saved_file:
                                                        st.success(f"💾 Binary data saved to `{saved_file}` for external analysis")
                                                        st.code(f"Analyze with: file {saved_file} && binwalk {saved_file} && strings {saved_file}")
                                        else:
                                            st.warning("No clear hidden content found with multi-bit LSB")
                                    except Exception as e:
                                        st.error(f"Multi-bit LSB extraction failed: {str(e)}")
                    
                    elif likelihood >= 0.2:
                        st.markdown("---")
                        st.info("💡 Low-moderate steganography likelihood detected. You can still try extraction methods using the Message Extractor tool.")
                
                # Add AI Assistant Analysis
                st.markdown("---")
                st.subheader("🤖 AI Investigation Assistant")
                
                # Initialize AI assistant
                try:
                    if AI_AVAILABLE and SteganographyAssistant is not None:
                        ai_assistant = SteganographyAssistant()
                    else:
                        raise ImportError("AI Assistant not available")
                    
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        if st.button("🔍 Get AI Analysis", help="Get expert analysis from AI assistant"):
                            with st.spinner("AI is analyzing the detection results..."):
                                try:
                                    # Get any extracted content for analysis
                                    sample_extracted = None
                                    if likelihood >= 0.4:
                                        try:
                                            quick_results = brute_force_decode(temp_path)
                                            successful = [r for r in quick_results if r.success and r.confidence > 0.3]
                                            if successful:
                                                sample_extracted = successful[0].data
                                        except:
                                            pass
                                    
                                    # Get AI analysis
                                    ai_analysis = ai_assistant.analyze_detection_results(
                                        detection_result, metadata, sample_extracted
                                    )
                                    
                                    # Display AI insights
                                    if ai_analysis:
                                        st.success("🤖 AI Analysis Complete!")
                                        
                                        # Summary
                                        st.write("**🎯 Summary:**")
                                        st.info(ai_analysis.get("summary", "Analysis complete"))
                                        
                                        # Technical Analysis
                                        if "technical_analysis" in ai_analysis:
                                            st.write("**🔬 Technical Analysis:**")
                                            st.write(ai_analysis["technical_analysis"])
                                        
                                        # Plain Language Explanation
                                        if "plain_language" in ai_analysis:
                                            st.write("**👤 In Simple Terms:**")
                                            st.write(ai_analysis["plain_language"])
                                        
                                        # Recommendations
                                        if "investigation_recommendations" in ai_analysis:
                                            st.write("**📋 Recommended Next Steps:**")
                                            for i, rec in enumerate(ai_analysis["investigation_recommendations"][:5], 1):
                                                st.write(f"{i}. {rec}")
                                        
                                        # Risk Assessment
                                        if "risk_assessment" in ai_analysis:
                                            risk_level = ai_analysis["risk_assessment"].lower()
                                            if "high" in risk_level:
                                                st.error(f"🔴 **Risk Level:** {ai_analysis['risk_assessment']}")
                                            elif "medium" in risk_level:
                                                st.warning(f"🟡 **Risk Level:** {ai_analysis['risk_assessment']}")
                                            else:
                                                st.success(f"🟢 **Risk Level:** {ai_analysis['risk_assessment']}")
                                        
                                        # Potential Techniques
                                        if "potential_techniques" in ai_analysis:
                                            st.write("**🎭 Suspected Techniques:**")
                                            techniques = ai_analysis["potential_techniques"]
                                            if isinstance(techniques, list):
                                                for tech in techniques[:3]:
                                                    st.write(f"• {tech}")
                                    
                                except Exception as e:
                                    st.error(f"AI analysis failed: {str(e)}")
                                    # Fallback to simple suggestions
                                    suggestions = get_investigation_suggestions(likelihood, detection_result.indicators if detection_result and hasattr(detection_result, 'indicators') else {})
                                    st.write("**💡 Investigation Suggestions:**")
                                    for suggestion in suggestions:
                                        st.write(f"• {suggestion}")
                    
                    with col2:
                        st.write("**Quick Insights:**")
                        suggestions = get_investigation_suggestions(likelihood, detection_result.indicators if detection_result and hasattr(detection_result, 'indicators') else {})
                        for suggestion in suggestions[:4]:
                            st.write(f"• {suggestion}")
                        
                        if likelihood >= 0.4:
                            st.write("")
                            st.markdown("**🎯 Priority Actions:**")
                            st.write("1. Extract hidden content")
                            st.write("2. Analyze extracted data")
                            st.write("3. Check file provenance")
                        
                except ImportError:
                    st.warning("🤖 AI Assistant requires OpenAI API access. Analysis features limited.")
                except Exception as e:
                    st.error(f"AI Assistant initialization failed: {str(e)}")
                        
                else:
                    st.write("No detailed detection data available")
                
                # Direct download buttons - no preview step
                st.markdown("---")
                st.subheader("📥 Download Analysis Reports")
                
                try:
                    # Generate comprehensive report data once
                    report_data = generate_detection_report(
                        uploaded_file.name,
                        detection_result,
                        metadata,
                        likelihood
                    )
                    
                    # Generate JSON report
                    report_json = json.dumps(report_data, indent=2, ensure_ascii=False)
                    
                    # Generate text report  
                    text_report = generate_text_report(
                        uploaded_file.name,
                        detection_result,
                        metadata,
                        likelihood
                    )
                    
                    # Show download buttons side by side (three columns)
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.download_button(
                            label="📄 Download JSON Report",
                            data=report_json,
                            file_name=f"steganography_analysis_{uploaded_file.name.rsplit('.', 1)[0]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                            mime="application/json",
                            help="Download detailed analysis results as JSON file",
                            use_container_width=True
                        )
                    
                    with col2:
                        st.download_button(
                            label="📝 Download Text Report", 
                            data=text_report,
                            file_name=f"steganography_analysis_{uploaded_file.name.rsplit('.', 1)[0]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                            mime="text/plain",
                            help="Download human-readable analysis report",
                            use_container_width=True
                        )
                    
                    with col3:
                        # Generate comprehensive HTML report
                        try:
                            # Collect all available analysis data
                            extracted_data = []
                            channel_analysis = None
                            
                            # Try to collect extraction results from session state if available
                            if hasattr(st.session_state, 'extraction_results'):
                                extracted_data = st.session_state.extraction_results
                            
                            # Try to collect channel analysis data from session state if available
                            if hasattr(st.session_state, 'channel_analysis'):
                                channel_analysis = st.session_state.channel_analysis
                            
                            # Generate HTML report with all available data
                            html_report = generate_comprehensive_html_report(
                                filename=uploaded_file.name,
                                detection_result=detection_result,
                                metadata=metadata,
                                likelihood=likelihood,
                                extracted_data=extracted_data,
                                channel_analysis=channel_analysis,
                                file_size=file_size,
                                entropy=entropy_value,
                                image_path=temp_path
                            )
                            
                            st.download_button(
                                label="🌐 Complete Analysis Report (HTML)",
                                data=html_report,
                                file_name=f"DEEP_ANAL_comprehensive_report_{uploaded_file.name.rsplit('.', 1)[0]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
                                mime="text/html",
                                help="Download comprehensive analysis report as viewable HTML file",
                                use_container_width=True,
                                type="primary"
                            )
                        except Exception as e:
                            st.error(f"Failed to generate HTML report: {str(e)}")
                            # Fallback HTML button without extra data
                            try:
                                html_report = generate_comprehensive_html_report(
                                    filename=uploaded_file.name,
                                    detection_result=detection_result,
                                    metadata=metadata,
                                    likelihood=likelihood,
                                    file_size=file_size,
                                    entropy=entropy_value,
                                    image_path=temp_path
                                )
                                
                                st.download_button(
                                    label="🌐 Complete Analysis Report (HTML)",
                                    data=html_report,
                                    file_name=f"DEEP_ANAL_comprehensive_report_{uploaded_file.name.rsplit('.', 1)[0]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
                                    mime="text/html",
                                    help="Download comprehensive analysis report as viewable HTML file",
                                    use_container_width=True,
                                    type="primary"
                                )
                            except Exception as fallback_error:
                                st.error(f"HTML report generation failed: {str(fallback_error)}")
                        
                except Exception as e:
                    st.error(f"Failed to generate report: {str(e)}")
            
            with tab3:
                st.subheader("File Metadata")
                
                # Display metadata in a clean format
                for key, value in metadata.items():
                    st.write(f"**{key}:** {value}")
            
            with tab4:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("String Analysis")
                    
                    try:
                        strings = extract_strings(temp_path)
                        if strings:
                            # Filter and clean strings for display
                            clean_strings = []
                            for s in strings:
                                s = s.strip()
                                if s and len(s) >= 4 and len(s) <= 100:
                                    # Filter out strings that are mostly the same character
                                    if len(set(s)) > 1:
                                        # Filter out HTML/CSS/code-like strings
                                        skip_string = False
                                        
                                        # Skip HTML tags and CSS
                                        if any(pattern in s.lower() for pattern in [
                                            '<div', '</div>', '<span', '</span>', 'style=', 'class=',
                                            'padding:', 'margin:', 'background-color:', 'border:', 'font-',
                                            'color:', 'rgba(', 'rgb(', '#00ffff', '#ff00ff',
                                            'word-break:', 'overflow:', 'position:', 'display:'
                                        ]):
                                            skip_string = True
                                        
                                        # Skip strings that look like CSS values or hex colors
                                        if s.startswith('#') and len(s) in [4, 7, 9]:  # hex colors
                                            skip_string = True
                                        
                                        # Skip strings that are mostly punctuation or symbols
                                        if len([c for c in s if c.isalnum()]) < len(s) * 0.5:
                                            skip_string = True
                                        
                                        # Skip strings with common code patterns
                                        if any(pattern in s for pattern in [
                                            '();', '{', '}', '<=', '>=', '&&', '||', 'px;', 'em;', 'rem;'
                                        ]):
                                            skip_string = True
                                        
                                        if not skip_string:
                                            clean_strings.append(s)
                            
                            if clean_strings:
                                st.write(f"**Found {len(clean_strings)} meaningful strings:**")
                                
                                # Create scrollable container with CSS
                                strings_container = f"""
                                <div style="
                                    height: 400px; 
                                    overflow-y: auto; 
                                    border: 1px solid #333; 
                                    padding: 10px; 
                                    background-color: rgba(0, 20, 40, 0.7);
                                    border-radius: 5px;
                                    font-family: monospace;
                                ">
                                """
                                
                                for i, string in enumerate(clean_strings[:100]):  # Limit to 100 strings
                                    # Escape HTML and truncate very long strings
                                    display_string = string.replace('<', '&lt;').replace('>', '&gt;')
                                    if len(display_string) > 80:
                                        display_string = display_string[:77] + "..."
                                    
                                    strings_container += f"""
                                    <div style="
                                        padding: 4px 8px; 
                                        margin: 2px 0; 
                                        background-color: rgba(0, 255, 255, 0.1);
                                        border-left: 3px solid #00ffff;
                                        font-size: 12px;
                                        color: #00ffff;
                                        word-break: break-all;
                                    ">
                                        <span style="color: #ff00ff; font-weight: bold;">{i+1:03d}:</span> {display_string}
                                    </div>
                                    """
                                
                                strings_container += "</div>"
                                st.markdown(strings_container, unsafe_allow_html=True)
                                
                                # Show statistics
                                st.caption(f"Showing top {min(100, len(clean_strings))} of {len(clean_strings)} strings found")
                            else:
                                st.info("No meaningful strings found after filtering")
                        else:
                            st.write("No readable strings extracted")
                    except Exception as e:
                        st.error(f"String analysis failed: {str(e)}")
                
                with col2:
                    st.subheader("ZSTEG Analysis")
                    
                    if file_type == 'png':
                        try:
                            with st.spinner("Running ZSTEG scan..."):
                                zsteg_output = run_zsteg(temp_path)
                            
                            if zsteg_output and zsteg_output.strip():
                                st.write("**ZSTEG Results:**")
                                st.code(zsteg_output, language="text")
                            else:
                                st.success("✓ No hidden data detected by ZSTEG")
                        except Exception as e:
                            st.error(f"ZSTEG analysis failed: {str(e)}")
                    else:
                        st.info("ZSTEG analysis only available for PNG files")
                
                st.subheader("Binary Analysis")
                
                col3, col4 = st.columns(2)
                
                with col3:
                    st.write("**File Structure (Binwalk)**")
                    try:
                        structure = analyze_file_structure(temp_path)
                        if structure and structure.strip():
                            st.code(structure, language="text")
                        else:
                            st.write("No embedded files detected")
                    except Exception as e:
                        st.error(f"Structure analysis failed: {str(e)}")
                
                with col4:
                    st.write("**Hex Dump (First 256 bytes)**")
                    try:
                        hex_dump = get_hex_dump(temp_path, 256)
                        if hex_dump:
                            # Display hex dump as plain text instead of HTML
                            st.code(hex_dump, language="text")
                        else:
                            st.write("No hex data available")
                    except Exception as e:
                        st.error(f"Hex analysis failed: {str(e)}")
            
            with tab5:
                st.subheader("🌈 RGB Channel Analysis")
                st.write("Analyzing individual color channels for hidden patterns and steganographic indicators...")
                
                try:
                    # Create channel analysis for each RGB channel
                    red_analysis = create_channel_analysis_visualization(temp_path, 'red')
                    green_analysis = create_channel_analysis_visualization(temp_path, 'green')
                    blue_analysis = create_channel_analysis_visualization(temp_path, 'blue')
                    
                    # Store channel analysis data in session state for HTML report
                    if red_analysis and green_analysis and blue_analysis:
                        st.session_state.channel_analysis = {
                            'red_stats': red_analysis.get('stats', {}),
                            'green_stats': green_analysis.get('stats', {}),
                            'blue_stats': blue_analysis.get('stats', {}),
                            'red_anomalies': red_analysis.get('anomalies', []),
                            'green_anomalies': green_analysis.get('anomalies', []),
                            'blue_anomalies': blue_analysis.get('anomalies', [])
                        }
                    
                    if red_analysis and green_analysis and blue_analysis:
                        # Display annotated anomaly analysis first if there are any anomalies
                        total_anomalies = len(red_analysis.get('anomalies', [])) + len(green_analysis.get('anomalies', [])) + len(blue_analysis.get('anomalies', []))
                        
                        if total_anomalies > 0:
                            st.subheader("🎯 Anomaly Detection Results")
                            st.write(f"Found **{total_anomalies}** potential steganographic anomalies across all channels.")
                            
                            # Display annotated plots for channels with anomalies
                            channels_with_anomalies = []
                            if len(red_analysis.get('anomalies', [])) > 0:
                                channels_with_anomalies.append(('Red', red_analysis))
                            if len(green_analysis.get('anomalies', [])) > 0:
                                channels_with_anomalies.append(('Green', green_analysis))
                            if len(blue_analysis.get('anomalies', [])) > 0:
                                channels_with_anomalies.append(('Blue', blue_analysis))
                            
                            for channel_name, analysis in channels_with_anomalies:
                                st.subheader(f"🔍 {channel_name} Channel Anomalies")
                                
                                # Display annotated plot
                                if 'annotated_plot' in analysis:
                                    st.plotly_chart(analysis['annotated_plot'], use_container_width=True)
                                
                                # Display anomaly details and recommendations
                                if 'annotations' in analysis:
                                    st.subheader(f"📋 {channel_name} Channel - Recommended Actions")
                                    
                                    for annotation in analysis['annotations']:
                                        with st.expander(f"🔴 Anomaly #{annotation['number']}: {annotation['type']} (Severity: {annotation['severity']})"):
                                            st.write(f"**Description:** {annotation['description']}")
                                            st.write("**Recommended Next Steps:**")
                                            for j, rec in enumerate(annotation['recommendations'], 1):
                                                st.write(f"{j}. {rec}")
                                            
                                            # Add severity-based urgency indicators
                                            severity_float = float(annotation['severity'])
                                            if severity_float >= 0.8:
                                                st.error("🚨 **HIGH PRIORITY** - Strong evidence of steganographic content")
                                            elif severity_float >= 0.6:
                                                st.warning("⚠️ **MEDIUM PRIORITY** - Suspicious patterns detected")
                                            else:
                                                st.info("ℹ️ **LOW PRIORITY** - Minor anomaly detected")
                            
                            st.markdown("---")
                        
                        # Channel comparison chart
                        comparison_plot = create_channel_comparison_plot(
                            red_analysis['stats'], 
                            green_analysis['stats'], 
                            blue_analysis['stats']
                        )
                        st.plotly_chart(comparison_plot, use_container_width=True)
                        
                        st.markdown("---")
                        
                        # Display individual channel analysis
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.subheader("🔴 Red Channel")
                            
                            # Display original channel data
                            st.write("**Original Channel:**")
                            st.image(red_analysis['original'], caption="Red Channel Data", use_container_width=True)
                            
                            # Display noise pattern
                            st.write("**Noise Pattern Analysis:**")
                            st.image(red_analysis['noise'], caption="Red Channel Noise", use_container_width=True)
                            
                            # Channel statistics
                            st.write("**Channel Statistics:**")
                            stats = red_analysis['stats']
                            st.write(f"- Mean: {stats['mean']:.2f}")
                            st.write(f"- Std Dev: {stats['std']:.2f}")
                            st.write(f"- Entropy: {stats['entropy']:.4f}")
                            st.write(f"- Histogram Peaks: {stats['histogram_peaks']}")
                        
                        with col2:
                            st.subheader("🟢 Green Channel")
                            
                            # Display original channel data
                            st.write("**Original Channel:**")
                            st.image(green_analysis['original'], caption="Green Channel Data", use_container_width=True)
                            
                            # Display noise pattern
                            st.write("**Noise Pattern Analysis:**")
                            st.image(green_analysis['noise'], caption="Green Channel Noise", use_container_width=True)
                            
                            # Channel statistics
                            st.write("**Channel Statistics:**")
                            stats = green_analysis['stats']
                            st.write(f"- Mean: {stats['mean']:.2f}")
                            st.write(f"- Std Dev: {stats['std']:.2f}")
                            st.write(f"- Entropy: {stats['entropy']:.4f}")
                            st.write(f"- Histogram Peaks: {stats['histogram_peaks']}")
                        
                        with col3:
                            st.subheader("🔵 Blue Channel")
                            
                            # Display original channel data
                            st.write("**Original Channel:**")
                            st.image(blue_analysis['original'], caption="Blue Channel Data", use_container_width=True)
                            
                            # Display noise pattern
                            st.write("**Noise Pattern Analysis:**")
                            st.image(blue_analysis['noise'], caption="Blue Channel Noise", use_container_width=True)
                            
                            # Channel statistics
                            st.write("**Channel Statistics:**")
                            stats = blue_analysis['stats']
                            st.write(f"- Mean: {stats['mean']:.2f}")
                            st.write(f"- Std Dev: {stats['std']:.2f}")
                            st.write(f"- Entropy: {stats['entropy']:.4f}")
                            st.write(f"- Histogram Peaks: {stats['histogram_peaks']}")
                        
                        st.markdown("---")
                        
                        # Analysis interpretation
                        st.subheader("📋 Channel Analysis Interpretation")
                        
                        # Calculate channel anomalies
                        entropies = [red_analysis['stats']['entropy'], green_analysis['stats']['entropy'], blue_analysis['stats']['entropy']]
                        entropy_variance = max(entropies) - min(entropies)
                        
                        if entropy_variance > 0.5:
                            st.warning(f"⚠️ **High entropy variance detected ({entropy_variance:.3f})**")
                            st.write("• Significant differences between channel entropies may indicate steganographic manipulation")
                            st.write("• Hidden data often affects specific color channels more than others")
                        else:
                            st.success(f"✅ **Normal entropy variance ({entropy_variance:.3f})**")
                            st.write("• Channel entropies are relatively uniform")
                            st.write("• No obvious signs of channel-specific manipulation")
                        
                        # Noise pattern analysis
                        st.write("**Noise Pattern Analysis:**")
                        st.write("• High-contrast noise patterns may indicate embedded data")
                        st.write("• Compare noise patterns between channels for anomalies")
                        st.write("• Regular patterns in noise suggest algorithmic data hiding")
                        
                    else:
                        st.error("Failed to analyze RGB channels - please try with a different image")
                
                except Exception as e:
                    st.error(f"Channel analysis failed: {str(e)}")
        else:
            # Non-image file - use binary file analysis
            st.info("📁 Non-image file detected. Running binary steganography analysis...")
            
            try:
                with st.spinner("Analyzing binary file for hidden data..."):
                    detection_result = analyze_binary_file_for_steganography(temp_path)
                likelihood = detection_result.likelihood
                likelihood_percentage = f"{likelihood*100:.1f}%"
                color = "#00ff00" if likelihood < 0.3 else "#ffff00" if likelihood < 0.6 else "#ff0000"
            except Exception as e:
                st.error(f"Binary analysis error: {str(e)}")
                likelihood = 0
                likelihood_percentage = "0.0%"
                color = "#00ff00"
                detection_result = None
            
            # Save to database
            if DB_AVAILABLE:
                try:
                    metadata_json = json.dumps(metadata)
                    save_analysis(uploaded_file.name, file_size, file_type, entropy_value, metadata_json)
                except Exception as e:
                    st.warning(f"Database save failed: {str(e)}")
        
            # Results display for binary files
            st.success(f"✓ Binary Analysis Complete: {uploaded_file.name}")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("File Size", f"{file_size} bytes")
            with col2:
                st.metric("Type", file_type.upper() if file_type else "BIN")
            with col3:
                st.metric("Entropy", f"{entropy_value:.4f}")
            with col4:
                st.metric("Stego Detection", likelihood_percentage, delta=None)
            
            # Show detection results
            if detection_result:
                st.markdown("---")
                st.subheader("🔍 Binary Steganography Detection")
                
                st.markdown(f"""
                <div style='background-color: rgba(0,255,255,0.1); padding: 20px; border-radius: 10px; border-left: 5px solid {color}'>
                    <h3 style='color: {color};'>Detection Likelihood: {likelihood_percentage}</h3>
                    <p>{detection_result.explanation}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Show indicators
                st.markdown("### 📊 Detection Indicators")
                for name, details in detection_result.indicators.items():
                    indicator_color = "#00ff00" if details['value'] < 0.3 else "#ffff00" if details['value'] < 0.6 else "#ff0000"
                    st.markdown(f"**{name}**: <span style='color:{indicator_color}'>{details['value']*100:.1f}%</span> (weight: {details['weight']})", unsafe_allow_html=True)
                
                if detection_result.techniques:
                    st.markdown("### 🔬 Suspected Techniques")
                    for technique in detection_result.techniques:
                        st.write(f"• {technique}")
            
            # Show basic visualizations that work for all files
            st.markdown("---")
            st.subheader("📊 Binary File Visualizations")
            
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Entropy Analysis**")
                entropy_plot = create_entropy_plot(entropy_value)
                st.plotly_chart(entropy_plot, use_container_width=True)
            
            with col2:
                st.write("**Byte Frequency**")
                bytes_values, frequencies = get_byte_frequency(temp_path)
                freq_plot = create_byte_frequency_plot(bytes_values, frequencies)
                st.plotly_chart(freq_plot, use_container_width=True)
            
            # Metadata and strings
            st.markdown("---")
            st.subheader("📄 File Metadata & Strings")
            
            with st.expander("📋 Metadata", expanded=False):
                if metadata:
                    for key, value in metadata.items():
                        st.write(f"**{key}**: {value}")
                else:
                    st.write("No metadata available")
            
            with st.expander("🔤 Extracted Strings", expanded=False):
                strings = extract_strings(temp_path, min_length=4)
                if strings:
                    st.text_area("Strings found:", "\n".join(strings[:100]), height=300)
                    if len(strings) > 100:
                        st.info(f"Showing first 100 of {len(strings)} strings found")
                else:
                    st.write("No readable strings found")
            
            st.warning("⚠️ Note: Image-specific visualizations (bitplanes, RGB analysis, etc.) are not available for non-image files.")
    
    except Exception as e:
        st.error(f"Critical error: {str(e)}")
    
    finally:
        # Cleanup
        try:
            os.unlink(temp_path)
        except:
            pass

else:
    # Instructions
    st.info("Upload ANY file type to begin steganography analysis")
    
    st.markdown("""
    ### What is DEEP ANAL?
    
    DEEP ANAL is an advanced steganography analysis tool that can detect hidden data in ANY file type using:
    
    - **LSB Analysis** - Examines least significant bits for patterns
    - **Statistical Tests** - Chi-square and entropy analysis  
    - **Histogram Analysis** - Detects frequency anomalies
    - **Metadata Inspection** - Checks for hidden information in headers
    - **String Extraction** - Finds readable text within binary data
    - **File Identification** - Uses magic bytes to identify true file types
    
    ### How to Use
    
    1. Upload ANY file (images, videos, documents, executables, archives, etc.)
    2. Wait for analysis to complete
    3. Review detection probability and detailed results
    4. Explore visualizations and extracted data
    """)