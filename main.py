import streamlit as st
import tempfile
import os
import json
import datetime
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

# Configure Streamlit page
st.set_page_config(
    page_title="DEEP ANAL: Steganography Analysis",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Clean cyberpunk theme
st.markdown("""
<style>
    .stApp {
        background-color: #000010;
        background-image: 
            radial-gradient(circle at 20% 90%, rgba(28, 0, 50, 0.4) 0%, transparent 20%),
            radial-gradient(circle at 80% 10%, rgba(0, 50, 90, 0.4) 0%, transparent 20%);
    }
    h1, h2, h3, h4, h5, h6 {
        color: #00ffff;
        font-family: monospace;
        text-shadow: 0 0 10px rgba(0, 255, 255, 0.5);
    }
    h1 {
        color: #ff00ff;
        text-shadow: 0 0 10px rgba(255, 0, 255, 0.5);
    }
</style>
""", unsafe_allow_html=True)

# Simple header
st.markdown("""
<div style="text-align: center; background-color: rgba(0, 10, 30, 0.7); padding: 20px; 
            border-radius: 10px; border: 1px solid #00ffff; margin-bottom: 20px;">
    <h1 style="color: #ff00ff; font-family: monospace; text-shadow: 0 0 10px rgba(255, 0, 255, 0.5);">
        DEEP ANAL
    </h1>
    <h3 style="color: #00ffff; font-family: monospace;">
        Steganography Analysis
    </h3>
</div>
""", unsafe_allow_html=True)

# File upload
uploaded_file = st.file_uploader(
    "Drop your file here",
    type=['png', 'jpg', 'jpeg'],
    help="Supported formats: PNG, JPEG"
)

if uploaded_file:
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
                else:
                    st.write("No detailed detection data available")
            
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
                            # Create word map visualization
                            strings_plot = create_strings_visualization(strings)
                            st.plotly_chart(strings_plot, use_container_width=True)
                            
                            # Display some extracted strings
                            st.write("**Sample Extracted Strings:**")
                            display_strings = [s for s in strings if len(s) > 3][:15]
                            for s in display_strings:
                                st.code(s)
                        else:
                            st.write("No readable strings found")
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