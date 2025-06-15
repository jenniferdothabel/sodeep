import streamlit as st
import tempfile
import os
import subprocess
from pathlib import Path
from utils.file_analysis import run_zsteg, extract_strings
from utils.stego_decoder import brute_force_decode

st.set_page_config(
    page_title="Hidden Message Extractor",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 Hidden Message Extractor")
st.write("**Specialized tool for extracting hidden messages from images**")

uploaded_file = st.file_uploader(
    "Upload image file",
    type=['png', 'jpg', 'jpeg'],
    help="Upload images that may contain hidden messages"
)

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        temp_path = tmp_file.name
    
    file_type = Path(uploaded_file.name).suffix.lower()[1:]
    
    st.success(f"Analyzing: {uploaded_file.name}")
    
    # Comprehensive extraction methods
    tab1, tab2, tab3, tab4 = st.tabs(["🔍 ZSTEG", "📝 Strings", "🛠️ Steghide", "🧬 Advanced"])
    
    with tab1:
        st.subheader("ZSTEG Analysis (PNG)")
        if file_type == 'png':
            with st.spinner("Running comprehensive ZSTEG scan..."):
                try:
                    # Run ZSTEG with all options
                    result = subprocess.run(
                        ['zsteg', '-a', str(temp_path)],
                        capture_output=True,
                        text=True,
                        timeout=60
                    )
                    
                    if result.stdout and result.stdout.strip():
                        st.write("**ZSTEG found potential hidden data:**")
                        st.code(result.stdout, language="text")
                        
                        # Look for readable text
                        lines = result.stdout.split('\n')
                        messages = []
                        for line in lines:
                            if any(keyword in line.lower() for keyword in ['text', 'ascii', 'message']):
                                messages.append(line)
                        
                        if messages:
                            st.write("**Potential text messages:**")
                            for msg in messages:
                                st.info(msg)
                    else:
                        st.write("No hidden data found with ZSTEG")
                        
                except Exception as e:
                    st.error(f"ZSTEG error: {str(e)}")
        else:
            st.info("ZSTEG analysis only works with PNG files")
    
    with tab2:
        st.subheader("String Extraction")
        with st.spinner("Extracting all readable strings..."):
            try:
                strings = extract_strings(temp_path, min_length=4)
                
                # Filter for potentially meaningful strings
                meaningful_strings = []
                for s in strings:
                    # Look for patterns that might be messages
                    if (len(s) > 10 and 
                        any(char.isalpha() for char in s) and
                        not s.startswith('/') and
                        'tmp' not in s.lower()):
                        meaningful_strings.append(s)
                
                if meaningful_strings:
                    st.write(f"**Found {len(meaningful_strings)} potentially meaningful strings:**")
                    for i, string in enumerate(meaningful_strings[:50]):  # Show first 50
                        if len(string) > 20:
                            st.write(f"{i+1}. `{string}`")
                else:
                    st.write("No meaningful text strings found")
                    
                # Show all strings in expandable section
                with st.expander(f"All extracted strings ({len(strings)} total)"):
                    for s in strings[:200]:  # Limit to first 200
                        st.code(s)
                        
            except Exception as e:
                st.error(f"String extraction error: {str(e)}")
    
    with tab3:
        st.subheader("Steghide Extraction")
        st.write("Attempting to extract data using steghide (for JPEG files)")
        
        if file_type in ['jpg', 'jpeg']:
            # Try common passwords
            passwords = ['', 'password', '123456', 'secret', 'navy', 'classified', 
                        uploaded_file.name.split('.')[0], 'dad', 'father']
            
            for pwd in passwords:
                try:
                    with st.spinner(f"Trying password: '{pwd if pwd else '(empty)'}'"):
                        result = subprocess.run(
                            ['steghide', 'extract', '-sf', str(temp_path), '-f'],
                            input=pwd,
                            capture_output=True,
                            text=True,
                            timeout=30
                        )
                        
                        if result.returncode == 0 and 'extracted' in result.stderr.lower():
                            st.success(f"✓ Successfully extracted with password: '{pwd if pwd else '(empty)'}'")
                            st.write(result.stderr)
                            break
                            
                except Exception as e:
                    continue
            else:
                st.write("No data extracted with common passwords")
                
                # Manual password entry
                manual_pwd = st.text_input("Try custom password:", type="password")
                if manual_pwd and st.button("Extract with custom password"):
                    try:
                        result = subprocess.run(
                            ['steghide', 'extract', '-sf', str(temp_path), '-f'],
                            input=manual_pwd,
                            capture_output=True,
                            text=True,
                            timeout=30
                        )
                        if result.returncode == 0:
                            st.success("✓ Extraction successful!")
                            st.write(result.stderr)
                        else:
                            st.error("Extraction failed with this password")
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
        else:
            st.info("Steghide works with JPEG files")
    
    with tab4:
        st.subheader("Advanced Analysis")
        
        # Binwalk for embedded files
        st.write("**Scanning for embedded files (Binwalk):**")
        try:
            result = subprocess.run(
                ['binwalk', str(temp_path)],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.stdout and result.stdout.strip():
                st.code(result.stdout, language="text")
                
                # Look for interesting findings
                if any(keyword in result.stdout.lower() for keyword in ['zip', 'encrypted', 'file', 'data']):
                    st.warning("⚠️ Binwalk found embedded files - this could contain hidden data!")
            else:
                st.write("No embedded files detected")
                
        except Exception as e:
            st.error(f"Binwalk error: {str(e)}")
        
        # Hex analysis for patterns
        st.write("**Hex pattern analysis:**")
        try:
            with open(temp_path, 'rb') as f:
                data = f.read(1024)  # First 1KB
                hex_data = data.hex()
                
                # Look for repeated patterns that might indicate hidden data
                if len(set(hex_data)) < len(hex_data) * 0.3:  # Low diversity might indicate patterns
                    st.warning("⚠️ Unusual hex patterns detected - possible steganography")
                else:
                    st.write("Normal hex data distribution")
                    
        except Exception as e:
            st.error(f"Hex analysis error: {str(e)}")
    
    # Cleanup
    try:
        os.unlink(temp_path)
    except:
        pass

else:
    st.info("Upload an image file to search for hidden messages")
    st.write("""
    This tool will attempt to extract hidden messages using:
    - ZSTEG (for PNG files)
    - String extraction with filtering
    - Steghide (for JPEG files) 
    - Binwalk for embedded files
    - Hex pattern analysis
    """)