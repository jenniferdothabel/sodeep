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

### 2. Steganography Detection (`utils/stego_detector.py`)
- **DetectionResult Class**: Structured container for analysis results
- **Multi-Indicator Analysis**: Combines multiple detection techniques
- **Weighted Scoring**: Configurable indicator weights for accuracy tuning
- **Confidence Calculation**: Statistical confidence metrics (46-47% sensitivity)
- **Technique Identification**: Suspected hiding method classification

### 3. Visualization Engine (`utils/visualizations.py`)
- **3D Entropy Plots**: Interactive cyberpunk-themed entropy visualization
- **Frequency Analysis**: Byte frequency distribution charts
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
- August 29, 2025: Added extensive image format support (TIFF, HEIC, BMP, WEBP, GIF) and ZIP batch upload
- June 15, 2025: Initial setup

## User Preferences

Preferred communication style: Simple, everyday language.