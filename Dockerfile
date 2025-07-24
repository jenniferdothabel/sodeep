# DEEP ANAL - Steganography Analysis Platform
# Multi-stage Docker build for optimized production deployment

FROM python:3.11-slim as base

# Install system dependencies for image processing and forensic tools
RUN apt-get update && apt-get install -y \
    libexif-dev \
    exiftool \
    binwalk \
    steghide \
    foremost \
    zsteg \
    ruby \
    ruby-dev \
    build-essential \
    libffi-dev \
    libssl-dev \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Install zsteg gem for Ruby
RUN gem install zsteg

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY local_requirements.txt .
RUN pip install --no-cache-dir -r local_requirements.txt

# Copy application files
COPY . .

# Create non-root user for security
RUN useradd -m -u 1000 deepanal && \
    chown -R deepanal:deepanal /app
USER deepanal

# Expose port
EXPOSE 5001

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:5001/_stcore/health || exit 1

# Start command
CMD ["streamlit", "run", "main.py", "--server.address=0.0.0.0", "--server.port=5001", "--server.headless=true"]