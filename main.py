import streamlit as st
import tempfile
import os
import json
from datetime import datetime
from pathlib import Path
from utils.file_analysis import (
    get_file_metadata, extract_strings, analyze_file_structure,
    calculate_entropy, get_byte_frequency, get_hex_dump, run_zsteg
)
from utils.visualizations import (
    create_entropy_plot, create_byte_frequency_plot, format_hex_dump,
    create_detailed_view, create_strings_visualization
)
from utils.database import (
    save_analysis, get_recent_analyses, get_analysis_by_id, DB_AVAILABLE
)
from utils.stego_detector import analyze_image_for_steganography
from utils.stego_decoder import (
    brute_force_decode, decode_lsb, decode_multi_bit_lsb, 
    try_steghide_extract, extract_metadata_hidden_data
)
from utils.ai_assistant import SteganographyAssistant, get_investigation_suggestions

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
st.set_page_config(
    page_title="DEEP ANAL: Steganography Analysis",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Simple clean header without HTML
st.title("🔍 DEEP ANAL")
st.subheader("Steganography Analysis Platform")

# Upload mode selection
upload_mode = st.radio(
    "Select Upload Mode:",
    ["🔍 Single File Analysis", "📊 Batch Scan (Multiple Files)"],
    horizontal=True
)

uploaded_file = None
if upload_mode == "🔍 Single File Analysis":
    # Single file upload
    uploaded_file = st.file_uploader(
        "Drop your file here",
        type=['png', 'jpg', 'jpeg', 'tiff', 'tif', 'bmp', 'webp', 'heic', 'heif', 'gif'],
        help="Supported formats: PNG, JPEG, TIFF, BMP, WEBP, HEIC, GIF"
    )
else:
    # Multi-file upload for batch processing
    st.write("**Option 1: Upload Multiple Images**")
    uploaded_files = st.file_uploader(
        "Drop multiple image files here",
        type=['png', 'jpg', 'jpeg', 'tiff', 'tif', 'bmp', 'webp', 'heic', 'heif', 'gif'],
        accept_multiple_files=True,
        help="Upload multiple images to quickly scan for steganography likelihood"
    )
    
    st.write("**Option 2: Upload ZIP Archive**")
    uploaded_zip = st.file_uploader(
        "Drop a ZIP file containing images",
        type=['zip'],
        help="Upload a ZIP archive containing images (PNG, JPEG, TIFF, BMP, WEBP, HEIC, GIF) for batch processing"
    )
    
    # Process ZIP file if uploaded
    if uploaded_zip:
        import zipfile
        import io
        
        try:
            with zipfile.ZipFile(io.BytesIO(uploaded_zip.getvalue())) as zip_ref:
                # Extract image files from ZIP
                image_files = []
                supported_extensions = ('.png', '.jpg', '.jpeg', '.tiff', '.tif', '.bmp', '.webp', '.heic', '.heif', '.gif',
                                       '.PNG', '.JPG', '.JPEG', '.TIFF', '.TIF', '.BMP', '.WEBP', '.HEIC', '.HEIF', '.GIF')
                
                for file_info in zip_ref.filelist:
                    if file_info.filename.endswith(supported_extensions) and not file_info.is_dir():
                        try:
                            file_data = zip_ref.read(file_info.filename)
                            # Create a file-like object that mimics uploaded_file
                            class ZipImageFile:
                                def __init__(self, name, data):
                                    self.name = name
                                    self.data = data
                                
                                def read(self):
                                    return self.data
                                
                                def getvalue(self):
                                    return self.data
                            
                            image_files.append(ZipImageFile(file_info.filename, file_data))
                        except Exception as e:
                            st.warning(f"Failed to extract {file_info.filename}: {str(e)}")
                
                if image_files:
                    st.success(f"📦 Extracted {len(image_files)} images from ZIP archive")
                    uploaded_files = image_files  # Use extracted files for batch processing
                else:
                    st.error("No supported image files found in ZIP archive")
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
                    
                    # Quick detection analysis
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
if upload_mode == "🔍 Single File Analysis" and uploaded_file:
    # Create temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        temp_path = tmp_file.name

    try:
        # Run analysis
        file_size = os.path.getsize(temp_path)
        file_type = Path(uploaded_file.name).suffix.lower()[1:]
        entropy_value = calculate_entropy(temp_path)
        metadata = get_file_metadata(temp_path)
        is_image = file_type in ['png', 'jpg', 'jpeg']
        
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
            tab1, tab2, tab3, tab4 = st.tabs(["📊 Visualizations", "🔍 Detection Details", "📄 Metadata", "🔬 Advanced"])
            
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
                                        
                                        if successful_results:
                                            st.success(f"✅ Found {len(successful_results)} potential hidden content(s)!")
                                            for i, result in enumerate(successful_results[:3]):  # Show top 3
                                                st.write(f"**Method {i+1}: {result.method}** (Confidence: {result.confidence:.2f})")
                                                
                                                # Display extracted data
                                                if result.data:
                                                    try:
                                                        # Try to decode as text first
                                                        text_data = result.data.decode('utf-8', errors='ignore')
                                                        if len(text_data.strip()) > 0 and all(ord(c) < 127 for c in text_data[:100]):
                                                            st.text_area(f"Extracted Text {i+1}:", text_data[:1000], height=100)
                                                        else:
                                                            st.write(f"**Binary data found:** {len(result.data)} bytes")
                                                            # Show hex preview
                                                            hex_preview = ' '.join(f'{b:02x}' for b in result.data[:32])
                                                            st.code(f"Hex preview: {hex_preview}{'...' if len(result.data) > 32 else ''}")
                                                            
                                                            # Save binary data to file for external analysis
                                                            saved_file = save_extracted_binary(result.data, result.method, i+1)
                                                            if saved_file:
                                                                st.success(f"💾 Binary data saved to `{saved_file}` for external analysis")
                                                                st.code(f"Analyze with: file {saved_file} && binwalk {saved_file} && strings {saved_file}")
                                                    except:
                                                        st.write(f"**Binary data found:** {len(result.data)} bytes")
                                                        # Save binary data to file for external analysis
                                                        saved_file = save_extracted_binary(result.data, result.method, i+1)
                                                        if saved_file:
                                                            st.success(f"💾 Binary data saved to `{saved_file}` for external analysis")
                                                            st.code(f"Analyze with: file {saved_file} && binwalk {saved_file} && strings {saved_file}")
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
                                                    else:
                                                        st.write(f"**Binary LSB data:** {len(best_result.data)} bytes")
                                                        # Save binary data to file for external analysis
                                                        saved_file = save_extracted_binary(best_result.data, "LSB", 2)
                                                        if saved_file:
                                                            st.success(f"💾 Binary data saved to `{saved_file}` for external analysis")
                                                            st.code(f"Analyze with: file {saved_file} && binwalk {saved_file} && strings {saved_file}")
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
                                                    else:
                                                        st.write(f"**Binary data extracted:** {len(result.data)} bytes")
                                                        # Save binary data to file for external analysis
                                                        saved_file = save_extracted_binary(result.data, "Steghide", 3)
                                                        if saved_file:
                                                            st.success(f"💾 Binary data saved to `{saved_file}` for external analysis")
                                                            st.code(f"Analyze with: file {saved_file} && binwalk {saved_file} && strings {saved_file}")
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
                                                        except:
                                                            st.write(f"**Binary data extracted:** {len(pass_result.data)} bytes")
                                                            # Save binary data to file for external analysis
                                                            saved_file = save_extracted_binary(pass_result.data, "Steghide_Password", 4)
                                                            if saved_file:
                                                                st.success(f"💾 Binary data saved to `{saved_file}` for external analysis")
                                                                st.code(f"Analyze with: file {saved_file} && binwalk {saved_file} && strings {saved_file}")
                                                else:
                                                    st.error("Password extraction failed")
                                    except Exception as e:
                                        st.error(f"Steghide extraction failed: {str(e)}")
                        
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
                                                except:
                                                    st.write(f"**Binary metadata:** {len(result.data)} bytes")
                                                    # Save binary data to file for external analysis
                                                    saved_file = save_extracted_binary(result.data, "Metadata", 5)
                                                    if saved_file:
                                                        st.success(f"💾 Binary data saved to `{saved_file}` for external analysis")
                                                        st.code(f"Analyze with: file {saved_file} && binwalk {saved_file} && strings {saved_file}")
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
                                                    else:
                                                        st.write(f"**Binary data:** {len(result.data)} bytes")
                                                        # Save binary data to file for external analysis
                                                        saved_file = save_extracted_binary(result.data, "Multi_LSB", 6)
                                                        if saved_file:
                                                            st.success(f"💾 Binary data saved to `{saved_file}` for external analysis")
                                                            st.code(f"Analyze with: file {saved_file} && binwalk {saved_file} && strings {saved_file}")
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
                    ai_assistant = SteganographyAssistant()
                    
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
                                    suggestions = get_investigation_suggestions(likelihood, detection_result.indicators if hasattr(detection_result, 'indicators') else {})
                                    st.write("**💡 Investigation Suggestions:**")
                                    for suggestion in suggestions:
                                        st.write(f"• {suggestion}")
                    
                    with col2:
                        st.write("**Quick Insights:**")
                        suggestions = get_investigation_suggestions(likelihood, detection_result.indicators if hasattr(detection_result, 'indicators') else {})
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
                
                # Add download button for results
                st.markdown("---")
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    if st.button("📥 Download Detection Report", help="Download comprehensive analysis results"):
                        try:
                            # Generate comprehensive report
                            report_data = generate_detection_report(
                                uploaded_file.name,
                                detection_result,
                                metadata,
                                likelihood
                            )
                            
                            # Convert to JSON
                            report_json = json.dumps(report_data, indent=2, ensure_ascii=False)
                            
                            # Offer download
                            st.download_button(
                                label="📄 Download JSON Report",
                                data=report_json,
                                file_name=f"steganography_analysis_{uploaded_file.name.rsplit('.', 1)[0]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                                mime="application/json",
                                help="Download detailed analysis results as JSON file"
                            )
                            
                            # Also offer text format
                            text_report = generate_text_report(
                                uploaded_file.name,
                                detection_result,
                                metadata,
                                likelihood
                            )
                            
                            st.download_button(
                                label="📝 Download Text Report",
                                data=text_report,
                                file_name=f"steganography_analysis_{uploaded_file.name.rsplit('.', 1)[0]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                                mime="text/plain",
                                help="Download human-readable analysis report"
                            )
                            
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
        else:
            st.error("Only PNG and JPEG images are supported for steganography analysis")
    
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
    st.info("Upload a PNG or JPEG image to begin steganography analysis")
    
    st.markdown("""
    ### What is DEEP ANAL?
    
    DEEP ANAL is an advanced steganography analysis tool that can detect hidden data in images using:
    
    - **LSB Analysis** - Examines least significant bits for patterns
    - **Statistical Tests** - Chi-square and entropy analysis  
    - **Histogram Analysis** - Detects frequency anomalies
    - **Metadata Inspection** - Checks for hidden information in headers
    - **String Extraction** - Finds readable text within binary data
    
    ### How to Use
    
    1. Upload a PNG or JPEG image
    2. Wait for analysis to complete
    3. Review detection probability and detailed results
    4. Explore visualizations and extracted data
    """)