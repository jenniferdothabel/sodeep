# DEEP ANAL - Local Installation Guide

## Quick Start

### Option 1: Docker Deployment (Recommended)

```bash
# Clone the repository
git clone <repository-url>
cd deep-anal

# Start with Docker Compose
docker-compose up -d

# Access the application
open http://localhost
```

### Option 2: Local Python Installation

```bash
# Clone the repository
git clone <repository-url>
cd deep-anal

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r local_requirements.txt

# Install system tools (Ubuntu/Debian)
sudo apt update
sudo apt install exiftool binwalk steghide foremost ruby ruby-dev
sudo gem install zsteg

# Start the application
streamlit run main.py --server.port=5001
```

---

## System Requirements

### Minimum Requirements
- **OS**: Linux (Ubuntu 20.04+), macOS 11+, Windows 10+
- **RAM**: 2GB
- **Storage**: 5GB free space
- **Python**: 3.8 or higher
- **Network**: Internet connection for initial setup

### Recommended Specifications
- **RAM**: 8GB or higher
- **CPU**: 4 cores or more
- **Storage**: 20GB SSD
- **Network**: Broadband connection

---

## Detailed Installation Instructions

### 1. System Dependencies

#### Ubuntu/Debian
```bash
sudo apt update
sudo apt install -y \
    python3 python3-pip python3-venv \
    exiftool binwalk steghide foremost \
    ruby ruby-dev build-essential \
    libffi-dev libssl-dev libpq-dev \
    postgresql postgresql-contrib

# Install zsteg
sudo gem install zsteg
```

#### macOS
```bash
# Install Homebrew if not already installed
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install dependencies
brew install python3 exiftool binwalk steghide foremost ruby postgresql
gem install zsteg
```

#### Windows
1. Install Python 3.8+ from [python.org](https://python.org)
2. Install WSL2 and Ubuntu for Linux tools
3. Follow Ubuntu instructions within WSL2

### 2. Database Setup (Optional but Recommended)

#### PostgreSQL Installation
```bash
# Start PostgreSQL service
sudo systemctl start postgresql
sudo systemctl enable postgresql

# Create database and user
sudo -u postgres psql
```

```sql
CREATE DATABASE deepanal;
CREATE USER deepanal WITH PASSWORD 'your_secure_password';
GRANT ALL PRIVILEGES ON DATABASE deepanal TO deepanal;
\q
```

#### Environment Variables
```bash
export DATABASE_URL="postgresql://deepanal:your_secure_password@localhost:5432/deepanal"
```

### 3. Application Installation

#### Download and Setup
```bash
# Clone repository
git clone <repository-url>
cd deep-anal

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install Python dependencies
pip install --upgrade pip
pip install -r local_requirements.txt
```

#### Configuration
```bash
# Create uploads directory
mkdir -p uploads

# Set permissions
chmod 755 uploads
```

### 4. Running the Application

#### Development Mode
```bash
# Start main application
streamlit run main.py --server.port=5001

# Optional: Start additional services
streamlit run debug_analysis.py --server.port=5002 &
streamlit run extract_hidden.py --server.port=5003 &
streamlit run create_test_images.py --server.port=5004 &
```

#### Production Mode with Docker
```bash
# Build and start services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

---

## Configuration Options

### Environment Variables
```bash
# Database connection (optional)
export DATABASE_URL="postgresql://user:pass@host:port/dbname"

# Application settings
export STREAMLIT_SERVER_PORT=5001
export STREAMLIT_SERVER_ADDRESS=0.0.0.0
export STREAMLIT_SERVER_HEADLESS=true

# Security settings
export STREAMLIT_SERVER_ENABLE_CORS=false
export STREAMLIT_SERVER_MAX_UPLOAD_SIZE=200
```

### Custom Configuration
Create `.streamlit/config.toml`:
```toml
[server]
port = 5001
address = "0.0.0.0"
headless = true
maxUploadSize = 200

[theme]
primaryColor = "#ff00ff"
backgroundColor = "#000010"
secondaryBackgroundColor = "#001020"
textColor = "#00ffff"
```

---

## Verification and Testing

### System Check
```bash
# Test system tools
exiftool --version
binwalk --help
steghide --version
zsteg --help

# Test Python dependencies
python3 -c "import streamlit, numpy, pandas, plotly; print('All dependencies OK')"
```

### Application Test
```bash
# Generate test images
python generate_test_images.py --clean --stego

# Run analysis test
python -c "
from utils.file_analysis import extract_strings
from utils.stego_detector import analyze_image_for_steganography
print('Testing clean image...')
result = analyze_image_for_steganography('clean_solid.png')
print(f'Clean image detection: {result.likelihood*100:.1f}%')
print('Testing steganographic image...')
result = analyze_image_for_steganography('stego_message.png')
print(f'Stego image detection: {result.likelihood*100:.1f}%')
"
```

### Expected Results
- Clean images: 10-30% detection rate
- Steganographic images: 70%+ detection rate
- Application accessible at http://localhost:5001

---

## Troubleshooting

### Common Issues

#### Port Already in Use
```bash
# Find process using port
sudo lsof -i :5001
# Kill process
sudo kill -9 <PID>
```

#### Permission Denied
```bash
# Fix file permissions
chmod +x *.py
chmod 755 utils/
```

#### Missing System Tools
```bash
# Verify installations
which exiftool binwalk steghide zsteg
# Reinstall if missing
sudo apt install --reinstall exiftool binwalk steghide
```

#### Database Connection Issues
```bash
# Check PostgreSQL status
sudo systemctl status postgresql
# Restart if needed
sudo systemctl restart postgresql
```

#### Python Dependency Conflicts
```bash
# Clean install
rm -rf venv
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r local_requirements.txt
```

### Getting Help
- Check application logs in terminal output
- Verify all system dependencies are installed
- Ensure Python virtual environment is activated
- Test with provided sample images first

---

## Production Deployment

### Security Considerations
- Change default database passwords
- Use HTTPS in production
- Implement proper authentication
- Regular security updates
- Firewall configuration

### Performance Optimization
- Use SSD storage for faster file processing
- Increase available RAM for large file analysis
- Configure database connection pooling
- Enable caching for repeated analyses

### Monitoring
- Set up application monitoring
- Database performance tracking
- Log aggregation and analysis
- Health check endpoints

---

## Uninstallation

### Remove Application
```bash
# Stop services
docker-compose down
# Remove files
rm -rf deep-anal/
# Remove virtual environment
rm -rf venv/
```

### Remove System Dependencies
```bash
# Ubuntu/Debian
sudo apt remove exiftool binwalk steghide foremost
sudo gem uninstall zsteg
```

---

**Support**: For installation assistance, please refer to the project documentation or contact the development team.