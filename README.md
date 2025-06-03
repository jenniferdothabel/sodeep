# DEEP ANAL - Advanced Steganography Analysis Tool

A comprehensive Streamlit-based steganography analysis platform designed for advanced image and file steganography research and visualization.

## Features

- **Advanced Steganography Detection**: AI-powered algorithms with 46-47% detection sensitivity
- **Interactive 3D Visualizations**: Cyberpunk-themed entropy and frequency analysis
- **Word Map Visualization**: Dynamic word cloud for extracted strings
- **Multi-Format Support**: PNG, JPEG, and other image formats
- **Database Integration**: PostgreSQL-backed analysis tracking
- **Real-time Analysis**: Instant feedback with probability indicators

## Quick Start

### Prerequisites
- Python 3.11+
- PostgreSQL database
- Required packages (see requirements below)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/deep-anal-steganography-scanner.git
cd deep-anal-steganography-scanner
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
export DATABASE_URL="your_postgresql_connection_string"
```

4. Run the application:
```bash
streamlit run main.py --server.address=0.0.0.0 --server.port=5001
```

## Dependencies

- streamlit
- numpy
- pandas
- pillow
- plotly
- psycopg2-binary
- scipy
- sqlalchemy

## Usage

1. **Upload Image**: Drag and drop or browse for PNG/JPEG files
2. **Analysis**: Automated scanning with multiple detection methods
3. **Visualization**: View entropy plots, frequency analysis, and string extraction
4. **Results**: Get probability scores and detailed analysis reports

## Visualizations

- **3D Entropy Plot**: Holographic visualization of data entropy
- **Byte Frequency Analysis**: Cyberpunk-styled frequency distribution
- **String Word Map**: Circular word cloud of extracted text
- **Hex Dump Viewer**: Color-coded binary data inspection

## Detection Methods

- LSB (Least Significant Bit) analysis
- Bit-pair correlation analysis
- Chi-square statistical testing
- Sample Pair Analysis (SPA)
- Metadata examination
- RGB channel correlation

## Technical Architecture

```
DEEP ANAL/
├── main.py                 # Main Streamlit application
├── utils/
│   ├── stego_detector.py   # Detection algorithms
│   ├── stego_decoder.py    # Decoding utilities
│   ├── file_analysis.py    # File processing
│   ├── visualizations.py   # 3D plotting and word maps
│   └── database.py         # PostgreSQL integration
├── assets/                 # Static resources
└── attached_assets/        # Sample test images
```

## Performance

- **Detection Sensitivity**: 46-47% (enhanced from baseline 23.3%)
- **Analysis Speed**: Real-time processing for images up to 200MB
- **Visualization**: Hardware-accelerated 3D rendering

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is open source. Please check individual component licenses for specific terms.

## Roadmap

- [ ] Video steganography support
- [ ] Mobile app deployment
- [ ] Enhanced ML detection models
- [ ] Side-by-side image comparison
- [ ] Batch processing capabilities

## Related Projects

- Aperi'solve: Online steganography tool
- Stegsolve: Java-based stego analysis
- Binwalk: Firmware analysis tool

## Contact

For questions or collaboration opportunities, please open an issue or reach out through GitHub.

---

**DEEP ANAL** - Because every bit matters in steganography analysis.