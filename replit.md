# DEEP ANAL: Steganography Analysis Tool

## Overview

DEEP ANAL is an advanced steganography analysis platform built with Streamlit that provides comprehensive image analysis capabilities to detect hidden data in image files. The application uses a combination of traditional forensic tools, statistical analysis, and AI-powered detection algorithms to identify potential steganographic content with cyberpunk-themed visualizations.

## System Architecture

### Frontend Architecture
- **Framework**: Streamlit for web-based interface
- **UI Theme**: Custom cyberpunk styling with neon colors and grid backgrounds
- **Visualization**: Plotly for interactive 3D plots and data visualization
- **Layout**: Wide layout with collapsible sidebar for optimal analysis viewing

### Backend Architecture
- **Language**: Python 3.11
- **Core Libraries**: NumPy, Pandas, PIL (Pillow), SciPy
- **File Processing**: Native Python with subprocess integration for external tools
- **Analysis Engine**: Custom steganography detection algorithms with configurable sensitivity

### Data Storage Solutions
- **Primary Database**: PostgreSQL 16 for analysis result persistence
- **ORM**: SQLAlchemy with declarative base models
- **Session Management**: SQLAlchemy sessionmaker with connection pooling
- **Fallback Mode**: Application continues without database if connection fails

## Key Components

### 1. File Analysis Module (`utils/file_analysis.py`)
- **Metadata Extraction**: Uses exiftool for comprehensive file metadata
- **String Extraction**: Binary string analysis with configurable minimum length
- **Entropy Calculation**: Statistical entropy analysis for randomness detection
- **Hex Analysis**: Raw byte-level examination capabilities
- **External Tool Integration**: Binwalk, steghide, foremost integration
- **Text Pattern Analysis**: Detects binary, base64, hexadecimal patterns
- **PGP/GPG Integration**: Automatic detection of cryptographic content

### 2. Steganography Detection (`utils/stego_detector.py`)
- **DetectionResult Class**: Structured container for analysis results
- **Multi-Indicator Analysis**: Combines multiple detection techniques
- **Weighted Scoring**: Configurable indicator weights for accuracy tuning
- **Confidence Calculation**: Statistical confidence metrics (46-47% sensitivity)
- **Technique Identification**: Suspected hiding method classification

### 2.5. PGP/GPG Analyzer (`utils/pgp_analyzer.py`)
- **Armor Block Detection**: Identifies PGP BEGIN/END markers for all block types
- **Key Analysis**: Detects public keys, private keys, and key IDs
- **Message Detection**: Finds encrypted messages and signed content
- **Signature Verification**: Analyzes digital signatures and checksums
- **Risk Assessment**: Automatic security risk level classification (critical/high/medium/low)
- **Forensic Recommendations**: Investigation guidance for detected cryptographic content

### 2.6. File Identifier (`utils/file_identifier.py`)
- **Magic Number Detection**: Uses libmagic/file command for accurate type identification
- **MIME Type Analysis**: Determines correct MIME type for any file
- **Signature Extraction**: Reads and analyzes file headers and magic bytes
- **Safety Assessment**: Checks for executables, oversized files, and encrypted content
- **Binary Classification**: Distinguishes between text and binary files
- **Universal Support**: Accepts and analyzes ALL file types (no restrictions)

### 3. Visualization Engine (`utils/visualizations.py`)
- **3D Entropy Plots**: Interactive cyberpunk-themed entropy visualization
- **Frequency Analysis**: Byte frequency distribution charts (2D/3D modes)
- **Bitplane Visualizer**: Extract and display all 24 bitplanes (8 per RGB channel)
- **RGB 3D Scatter Plot**: Map pixels into 3D color space with density smoothing
- **Entropy Terrain Map**: Block-based Shannon entropy heightmap visualization
- **Segment Structure Mapper**: Parse and visualize file format structure (PNG/JPEG/generic)
- **Word Cloud Generation**: String extraction visualization
- **Color-Coded Results**: Visual feedback based on detection confidence
- **Responsive Design**: Adaptive layouts for different screen sizes

### 4. Database Layer (`utils/database.py`)
- **AnalysisResult Model**: Stores file metadata, entropy, and detection results
- **Connection Management**: Automatic fallback handling for database unavailability
- **Session Lifecycle**: Proper session creation and cleanup
- **JSON Metadata Storage**: Flexible metadata storage with TEXT field

## Data Flow

1. **File Upload**: User uploads image through Streamlit file uploader
2. **Temporary Storage**: File saved to temporary location for processing
3. **Parallel Analysis**: Multiple analysis modules process file simultaneously:
   - Metadata extraction via exiftool
   - Entropy calculation using statistical methods
   - String extraction with configurable parameters
   - Steganography detection algorithms
4. **Result Aggregation**: All analysis results combined into unified report
5. **Visualization Generation**: Interactive plots created based on analysis data
6. **Database Storage**: Results optionally saved to PostgreSQL for tracking
7. **UI Presentation**: Results displayed with cyberpunk-themed interface

## External Dependencies

### System Tools
- **exiftool**: EXIF metadata extraction
- **binwalk**: Binary analysis and embedded file detection
- **steghide**: Steganography tool integration
- **foremost**: File carving capabilities
- **strings**: ASCII string extraction

### Python Libraries
- **streamlit**: Web application framework
- **plotly**: Interactive visualization library
- **psycopg2-binary**: PostgreSQL adapter
- **sqlalchemy**: Database ORM
- **scipy**: Scientific computing for statistical analysis
- **numpy/pandas**: Data manipulation and analysis

### System Packages
- **PostgreSQL 16**: Database server
- **Python 3.11**: Runtime environment
- **Image processing libraries**: freetype, libjpeg, libpng, libwebp

## Deployment Strategy

### Replit Configuration
- **Deployment Target**: Autoscale for dynamic resource allocation
- **Port Configuration**: Multiple ports (5000, 5001, 5002) for parallel services
- **Environment**: Nix-based with stable-24_05 channel
- **Package Management**: UV for Python dependency resolution

### Application Variants
- **Main Application** (`main.py`): Full-featured analysis interface (Port 5001)
- **Minimal Version** (`minimal.py`): Lightweight testing interface (Port 5000)
- **Debug Interface** (`debug_analysis.py`): Development debugging tools (Port 5002)

### Workflow Management
- **Parallel Execution**: Main app and debug interface run simultaneously
- **Health Monitoring**: Port-based service availability checking
- **Graceful Degradation**: Database-optional operation for reliability

## Changelog
- October 27, 2025: **Removed all file type restrictions** - application now accepts ANY file type with intelligent file identification using magic numbers. Added comprehensive File Identifier module using libmagic for MIME type detection, signature extraction, and safety assessment. Enhanced UI with file type display, magic bytes analysis, and security warnings.
- October 27, 2025: Added PGP/GPG workflow analysis module for detecting encrypted messages, public/private keys, and signatures in extracted content. Features automatic risk assessment, key ID extraction, and forensic investigation recommendations.
- October 24, 2025: Added 5 advanced visualization modules:
  1. Byte Frequency Upgrade (2D heatmap / 3D bar graph toggle)
  2. Bitplane Visualizer (24-layer analysis for LSB/MSB detection)
  3. RGB 3D Scatter Plot (color space distribution with density analysis)
  4. Entropy Terrain Map (block-based Shannon entropy heightmap)
  5. Segment Structure Mapper (PNG chunks, JPEG markers, file format parsing)
- October 24, 2025: Updated AI assistant to use GPT-4o model
- August 30, 2025: Added OCR text extraction with steganographic pattern analysis and XOR decoding capabilities with automatic key detection
- August 29, 2025: Added extensive image format support (TIFF, HEIC, BMP, WEBP, GIF), ZIP batch upload, and comprehensive video format support (MP4, AVI, MOV, WMV, FLV, MKV, WEBM)
- June 15, 2025: Initial setup

## User Preferences

Preferred communication style: Simple, everyday language.