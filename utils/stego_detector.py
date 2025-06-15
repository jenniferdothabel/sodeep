"""
Steganography detection module.
Provides functionality to analyze images and determine the likelihood of hidden data.
"""

import numpy as np
from PIL import Image
import subprocess
import tempfile
import os
import re
import struct
from scipy import stats
import random

class DetectionResult:
    """Container for detection results."""
    def __init__(self):
        self.likelihood = 0.0  # Overall likelihood of hidden data (0-1)
        self.indicators = {}  # Individual indicator results
        self.suspicious_regions = []  # Areas of the image that might contain hidden data
        self.explanation = ""  # Human-readable explanation
        self.techniques = []  # Suspected hiding techniques
    
    def add_indicator(self, name, value, weight=1.0):
        """Add a new detection indicator."""
        self.indicators[name] = {
            "value": value,
            "weight": weight
        }
    
    def calculate_overall_likelihood(self):
        """Calculate the overall likelihood based on indicators."""
        if not self.indicators:
            return 0.0
        
        total_weight = sum(ind["weight"] for ind in self.indicators.values())
        weighted_sum = sum(ind["value"] * ind["weight"] for ind in self.indicators.values())
        
        if total_weight > 0:
            self.likelihood = weighted_sum / total_weight
        
        return self.likelihood
    
    def get_formatted_likelihood(self):
        """Get a formatted string with the likelihood percentage."""
        return f"{self.likelihood * 100:.1f}%"
    
    def get_color_code(self):
        """Get a color code based on the likelihood."""
        if self.likelihood < 0.3:
            return "#00ff00"  # Green - low likelihood
        elif self.likelihood < 0.7:
            return "#ffff00"  # Yellow - medium likelihood
        else:
            return "#ff0000"  # Red - high likelihood
    
    def generate_explanation(self):
        """Generate a human-readable explanation of the results."""
        if self.likelihood < 0.1:
            main_finding = "No significant indicators of steganography detected. The image appears normal."
        elif self.likelihood < 0.3:
            main_finding = "Some minor irregularities detected, but they could be due to normal image processing."
        elif self.likelihood < 0.6:
            main_finding = "Several indicators suggest possible hidden data. The image shows patterns that may be consistent with steganographic techniques."
        elif self.likelihood < 0.8:
            main_finding = "High likelihood of hidden data detected. Multiple indicators suggest steganographic content."
        else:
            main_finding = "Very strong evidence of hidden data. The image exhibits clear signs of steganographic manipulation."
        
        # Add details about the specific indicators
        indicator_details = []
        for name, details in self.indicators.items():
            strength = "strong" if details["value"] > 0.7 else "moderate" if details["value"] > 0.4 else "weak"
            indicator_details.append(f"{name} shows {strength} indication ({details['value']*100:.1f}%)")
        
        # Store the main finding separate from the details
        self.explanation = main_finding
        self.main_finding = main_finding
        self.detailed_findings = indicator_details
        
        return self.explanation

# Main detection functions
def analyze_image_for_steganography(image_path):
    """
    Analyze an image for signs of steganography.
    
    Args:
        image_path: Path to the image file
    
    Returns:
        DetectionResult object with likelihood and explanations
    """
    result = DetectionResult()
    
    # Open the image
    try:
        img = Image.open(image_path)
        if img.mode != 'RGB' and img.mode != 'RGBA':
            img = img.convert('RGB')
        
        # Convert to numpy array and resize if too large to prevent hanging
        pixels = np.array(img)
        
        # Optimize for large images - resize if larger than 1000x1000
        if pixels.shape[0] > 1000 or pixels.shape[1] > 1000:
            # Calculate new size maintaining aspect ratio
            max_dim = max(pixels.shape[0], pixels.shape[1])
            scale = 1000 / max_dim
            new_height = int(pixels.shape[0] * scale)
            new_width = int(pixels.shape[1] * scale)
            
            # Resize image
            img_resized = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            pixels = np.array(img_resized)
        
        # Run optimized detection methods
        
        # 1. Statistical analysis of LSB (optimized)
        lsb_likelihood = detect_lsb_steganography(pixels)
        result.add_indicator("LSB Analysis", lsb_likelihood, weight=1.5)
        
        # 2. Histogram analysis (fast)
        histogram_likelihood = analyze_histogram(pixels)
        result.add_indicator("Histogram Analysis", histogram_likelihood, weight=1.2)
        
        # 3. Simplified noise analysis
        noise_likelihood = analyze_noise_patterns(pixels)
        result.add_indicator("Noise Analysis", noise_likelihood, weight=1.0)
        
        # 4. Chi-square analysis (sampled for speed)
        chi_square_likelihood = chi_square_test(pixels)
        result.add_indicator("Chi-Square Test", chi_square_likelihood, weight=1.3)
        
        # 5. Metadata analysis
        metadata_likelihood = analyze_metadata(image_path)
        result.add_indicator("Metadata Analysis", metadata_likelihood, weight=0.8)
        
        # 6. Sample pair analysis
        sample_pair_likelihood = sample_pair_analysis(pixels)
        result.add_indicator("Sample Pair Analysis", sample_pair_likelihood, weight=1.1)
        
        # 7. RGB correlation
        rgb_correlation_likelihood = analyze_rgb_correlation(pixels)
        result.add_indicator("RGB Correlation", rgb_correlation_likelihood, weight=1.0)
        
        # Calculate overall likelihood
        result.calculate_overall_likelihood()
        
        # Generate explanation
        result.generate_explanation()
        
        # Determine potential techniques
        result.techniques = determine_potential_techniques(result)
        
    except Exception as e:
        result.explanation = f"Error analyzing image: {str(e)}"
        result.likelihood = 0.0
    
    return result

def detect_lsb_steganography(pixels):
    """
    Detect LSB steganography by analyzing the statistical properties of the least significant bits.
    
    Args:
        pixels: Numpy array of pixel values
    
    Returns:
        Likelihood of LSB steganography (0-1)
    """
    # Sample pixels for large images to prevent hanging
    max_pixels = 100000  # Limit analysis to 100k pixels for speed
    total_pixels = pixels.shape[0] * pixels.shape[1]
    
    if total_pixels > max_pixels:
        # Sample pixels randomly
        indices = np.random.choice(total_pixels, max_pixels, replace=False)
        flat_pixels = pixels.reshape(-1, 3)
        sampled_pixels = flat_pixels[indices]
        sampled_pixels = sampled_pixels.reshape(-1, 1, 3)
    else:
        sampled_pixels = pixels
    
    # Extract LSBs from each channel
    lsb_r = sampled_pixels[:, :, 0].flatten() % 2
    lsb_g = sampled_pixels[:, :, 1].flatten() % 2
    lsb_b = sampled_pixels[:, :, 2].flatten() % 2
    
    # Calculate bias from expected distribution (should be ~0.5 for random)
    r_bias = abs(np.mean(lsb_r) - 0.5) * 2
    g_bias = abs(np.mean(lsb_g) - 0.5) * 2
    b_bias = abs(np.mean(lsb_b) - 0.5) * 2
    
    # Simplified run analysis for speed
    r_runs = min(count_runs(lsb_r[:10000]) / min(len(lsb_r), 10000), 1.0)  # Limit to 10k samples
    g_runs = min(count_runs(lsb_g[:10000]) / min(len(lsb_g), 10000), 1.0)
    b_runs = min(count_runs(lsb_b[:10000]) / min(len(lsb_b), 10000), 1.0)
    
    # Fast entropy calculation
    r_entropy = calculate_entropy(lsb_r[:5000])  # Limit entropy calc to 5k samples
    g_entropy = calculate_entropy(lsb_g[:5000])
    b_entropy = calculate_entropy(lsb_b[:5000])
    
    # Also analyze the distribution of pairs of adjacent bits
    # This can detect more sophisticated steganography methods
    pair_analysis_r = analyze_bit_pairs(lsb_r[:5000])  # Limit for speed
    pair_analysis_g = analyze_bit_pairs(lsb_g[:5000])  # Limit for speed
    pair_analysis_b = analyze_bit_pairs(lsb_b[:5000])  # Limit for speed
    
    # Check for patterns in different bit planes (not just LSB) - use sampled pixels
    second_bit_plane_r = (sampled_pixels[:, :, 0] // 2) % 2
    second_bit_plane_g = (sampled_pixels[:, :, 1] // 2) % 2
    second_bit_plane_b = (sampled_pixels[:, :, 2] // 2) % 2
    
    # Calculate correlation between LSB and 2nd bit plane
    # Low correlation might indicate hidden data
    r_correlation = np.corrcoef(lsb_r.flatten(), second_bit_plane_r.flatten())[0, 1]
    g_correlation = np.corrcoef(lsb_g.flatten(), second_bit_plane_g.flatten())[0, 1]
    b_correlation = np.corrcoef(lsb_b.flatten(), second_bit_plane_b.flatten())[0, 1]
    
    # Convert correlations to indicators (lower correlation = higher likelihood)
    r_corr_indicator = 1 - abs(r_correlation)
    g_corr_indicator = 1 - abs(g_correlation)
    b_corr_indicator = 1 - abs(b_correlation)
    
    # Combine metrics - lower bias, more runs, higher entropy, and lower bit-plane correlation
    # indicate potential hidden data
    
    # Normalize entropy to 0-1 range (max entropy for binary is 1.0)
    r_entropy_norm = r_entropy 
    g_entropy_norm = g_entropy
    b_entropy_norm = b_entropy
    
    # Recalibrated weights for more realistic detection
    # High entropy and low bias are normal for photographs, so reduce their weight
    r_indicator = (1 - r_bias) * 0.2 + r_entropy_norm * 0.15 + r_runs * 0.3 + pair_analysis_r * 0.2 + r_corr_indicator * 0.15
    g_indicator = (1 - g_bias) * 0.2 + g_entropy_norm * 0.15 + g_runs * 0.3 + pair_analysis_g * 0.2 + g_corr_indicator * 0.15
    b_indicator = (1 - b_bias) * 0.2 + b_entropy_norm * 0.15 + b_runs * 0.3 + pair_analysis_b * 0.2 + b_corr_indicator * 0.15
    
    # Take the average instead of max to be less aggressive
    lsb_likelihood = (r_indicator + g_indicator + b_indicator) / 3
    
    # Reduce sensitivity significantly for more realistic results
    lsb_likelihood = scale_likelihood(lsb_likelihood, sensitivity=1.2)
    
    return lsb_likelihood

def analyze_bit_pairs(bits):
    """
    Analyze the distribution of pairs of adjacent bits.
    Unusual pair distributions may indicate steganography.
    
    Args:
        bits: Numpy array of bit values (0 or 1)
    
    Returns:
        Likelihood score based on bit pair analysis (0-1)
    """
    # Count the frequency of each pair type: 00, 01, 10, 11
    pairs = np.zeros(4)
    for i in range(len(bits) - 1):
        pair_value = bits[i] * 2 + bits[i+1]
        pairs[int(pair_value)] += 1
    
    # Normalize to get distribution
    total = np.sum(pairs)
    if total > 0:
        pairs = pairs / total
    
    # For true random data, each pair should be roughly equally likely (~0.25)
    # Calculate deviation from expected distribution
    expected = np.array([0.25, 0.25, 0.25, 0.25])
    deviation = np.sum(np.abs(pairs - expected)) / 2  # Normalize to [0,1]
    
    # Higher deviation means more suspicious
    return deviation

def analyze_histogram(pixels):
    """
    Analyze image histogram for signs of manipulation.
    
    Args:
        pixels: Numpy array of pixel values
    
    Returns:
        Likelihood based on histogram analysis (0-1)
    """
    # Analyze each channel separately
    likelihoods = []
    
    for channel in range(3):  # R, G, B
        # Get channel data
        channel_data = pixels[:, :, channel].flatten()
        
        # Create histogram (256 bins for 8-bit values)
        hist, _ = np.histogram(channel_data, bins=256, range=(0, 256))
        
        # Calculate the difference between adjacent histogram bins
        # Steganography often causes unusual patterns in these differences
        diffs = np.abs(hist[1:] - hist[:-1])
        
        # Calculate statistics on the differences
        mean_diff = np.mean(diffs)
        std_diff = np.std(diffs)
        
        # Calculate the number of "peaks" and "valleys"
        # Compare each bin with its neighbors
        peaks = 0
        for i in range(1, 255):
            if hist[i] > hist[i-1] and hist[i] > hist[i+1]:
                peaks += 1
        
        # Normalize peaks (typical images might have 5-20 peaks)
        normalized_peaks = min(peaks / 30, 1.0)
        
        # Calculate "evenness" of histogram
        # Use Gini coefficient as a measure of inequality
        hist_sorted = np.sort(hist)
        cumulative = np.cumsum(hist_sorted)
        cumulative = cumulative / cumulative[-1]  # Normalize
        gini = (np.trapz(np.linspace(0, 1, 256), cumulative) - 0.5) * 2
        
        # Combine metrics
        # Higher peaks, lower Gini (more even distribution) suggest steganography
        channel_likelihood = normalized_peaks * 0.4 + (1 - gini) * 0.6
        likelihoods.append(channel_likelihood)
    
    # Take the maximum likelihood across channels
    histogram_likelihood = max(likelihoods)
    
    # Scale the result
    histogram_likelihood = scale_likelihood(histogram_likelihood)
    
    return histogram_likelihood

def analyze_noise_patterns(pixels):
    """
    Analyze noise patterns in the image for signs of steganography.
    
    Args:
        pixels: Numpy array of pixel values
    
    Returns:
        Likelihood based on noise analysis (0-1)
    """
    # Apply noise extraction filter
    # A simple way is to use high-pass filtering
    noise = np.zeros_like(pixels[:, :, 0:3], dtype=float)
    
    for channel in range(3):
        # Extract channel
        img_channel = pixels[:, :, channel].astype(float)
        
        # Calculate local average (3x3 window)
        from scipy.ndimage import uniform_filter
        local_avg = uniform_filter(img_channel, size=3)
        
        # Extract noise component
        noise[:, :, channel] = img_channel - local_avg
    
    # Calculate noise statistics
    noise_flat = noise.reshape(-1, 3)
    
    # Calculate standard deviation of noise
    noise_std = np.std(noise_flat, axis=0)
    
    # Calculate noise correlation between pixels
    # Correlated noise is more likely in normal images
    # Uncorrelated noise might indicate steganography
    
    # Sample a subset of pixels for correlation calculation (for performance)
    sample_size = min(10000, noise_flat.shape[0])
    indices = np.random.choice(noise_flat.shape[0], sample_size, replace=False)
    noise_sample = noise_flat[indices]
    
    # Calculate correlation matrix
    correlation = np.corrcoef(noise_sample.T)
    
    # Average absolute correlation between channels
    avg_correlation = 0
    count = 0
    for i in range(3):
        for j in range(i+1, 3):
            avg_correlation += abs(correlation[i, j])
            count += 1
    
    if count > 0:
        avg_correlation /= count
    
    # Lower correlation suggests steganography
    correlation_indicator = 1 - avg_correlation
    
    # Check noise distribution
    # Calculate how close the noise is to a normal distribution
    # For each channel
    normality_scores = []
    for channel in range(3):
        # Sample data for speed
        channel_sample = noise_flat[:, channel][indices]
        
        # Normalize data
        normalized_data = (channel_sample - np.mean(channel_sample)) / np.std(channel_sample)
        
        # Perform Shapiro-Wilk test for normality
        # Lower p-value indicates deviation from normality
        try:
            from scipy import stats
            _, p_value = stats.shapiro(normalized_data[:1000])  # Limit sample size for performance
            normality_scores.append(1 - p_value)  # Higher score = less normal = more suspicious
        except:
            # Fallback if shapiro fails
            normality_scores.append(0.5)
    
    # Average normality score
    avg_normality = np.mean(normality_scores)
    
    # Combine metrics
    noise_likelihood = correlation_indicator * 0.6 + avg_normality * 0.4
    
    # Scale the result
    noise_likelihood = scale_likelihood(noise_likelihood)
    
    return noise_likelihood

def chi_square_test(pixels):
    """
    Perform chi-square test to detect LSB steganography.
    
    Args:
        pixels: Numpy array of pixel values
    
    Returns:
        Likelihood based on chi-square test (0-1)
    """
    chi_square_values = []
    
    # Process each color channel
    for channel in range(3):
        # Get channel data
        channel_data = pixels[:, :, channel].flatten()
        
        # Expected frequencies: pairs of values that differ only in LSB should be roughly equal
        expected_pairs = {}
        observed_pairs = {}
        
        for i in range(0, 255, 2):
            # Pairs that differ only in LSB: (i, i+1)
            pair_key = i // 2
            expected_pairs[pair_key] = 0
            observed_pairs[pair_key] = [0, 0]  # [count for i, count for i+1]
        
        # Count observed frequencies
        for pixel_value in channel_data:
            pair_key = (pixel_value // 2)
            if pair_key in observed_pairs:
                idx = pixel_value % 2
                observed_pairs[pair_key][idx] += 1
        
        # Calculate expected frequencies (average of pair)
        for pair_key, counts in observed_pairs.items():
            expected_pairs[pair_key] = (counts[0] + counts[1]) / 2
        
        # Calculate chi-square statistic
        chi_square = 0
        for pair_key, expected in expected_pairs.items():
            if expected > 0:  # Avoid division by zero
                chi_square += (observed_pairs[pair_key][0] - expected)**2 / expected
                chi_square += (observed_pairs[pair_key][1] - expected)**2 / expected
        
        # Normalize chi-square value
        chi_square /= len(channel_data)
        
        # Convert to p-value
        # Lower p-value indicates discrepancy between observed and expected
        try:
            from scipy import stats
            df = len(expected_pairs) - 1  # Degrees of freedom
            p_value = 1 - stats.chi2.cdf(chi_square, df)
            
            # Lower p-value means more likely to have hidden data
            chi_square_values.append(1 - p_value)
        except:
            # Fallback if chi2 calculation fails
            chi_square_values.append(0.5)
    
    # Take the maximum value as our indicator
    chi_square_likelihood = max(chi_square_values)
    
    # Scale the result
    chi_square_likelihood = scale_likelihood(chi_square_likelihood)
    
    return chi_square_likelihood

def analyze_metadata(image_path):
    """
    Analyze image metadata for signs of steganography.
    
    Args:
        image_path: Path to the image file
    
    Returns:
        Likelihood based on metadata analysis (0-1)
    """
    try:
        # Run exiftool to extract metadata
        with tempfile.NamedTemporaryFile(suffix='.txt') as tmp_file:
            cmd = ["exiftool", image_path, "-a", "-u", "-g1"]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                return 0.4  # Neutral value if exiftool fails
            
            metadata_text = result.stdout
            
            # Look for suspicious indicators in metadata
            suspicious_indicators = 0
            total_indicators = 6  # Number of checks we're performing
            
            # 1. Check for unusual or non-standard metadata fields
            unusual_fields = ["UserComment", "ImageUniqueID", "OwnerName", "Comment", "XMP"]
            for field in unusual_fields:
                if field in metadata_text:
                    suspicious_indicators += 1
            
            # 2. Check for unusually large metadata
            if len(metadata_text) > 2000:  # Arbitrary threshold
                suspicious_indicators += 1
            
            # 3. Check for binary or encoded data in text fields
            binary_patterns = [
                r'\\x[0-9a-fA-F]{2}',  # Hex escape sequences
                r'[A-Za-z0-9+/=]{20,}',  # Possible base64
                r'(?:\x00){3,}'  # Null byte sequences
            ]
            
            for pattern in binary_patterns:
                if re.search(pattern, metadata_text):
                    suspicious_indicators += 1
                    break
            
            # 4. Check for modification timestamps that don't align
            timestamps = re.findall(r'Date/Time.*?: (.*?)$', metadata_text, re.MULTILINE)
            if len(timestamps) > 1:
                timestamp_set = set(timestamps)
                if len(timestamp_set) > 1:
                    suspicious_indicators += 0.5
            
            # 5. Check for multiple tool traces
            editing_tools = re.findall(r'Software.*?: (.*?)$', metadata_text, re.MULTILINE)
            if len(editing_tools) > 1:
                suspicious_indicators += 0.5
            
            # 6. Check for steganography tool signatures
            stego_tools = ["outguess", "steghide", "stegdetect", "jsteg", "f5", "steganography"]
            for tool in stego_tools:
                if tool.lower() in metadata_text.lower():
                    suspicious_indicators += 2  # Strong indicator
                    break
            
            # Calculate likelihood based on indicators
            likelihood = suspicious_indicators / total_indicators
            
            # Scale the result
            likelihood = scale_likelihood(likelihood, sensitivity=1.2)
            
            return likelihood
    except Exception as e:
        return 0.4  # Neutral value if analysis fails

def sample_pair_analysis(pixels):
    """
    Perform Sample Pair Analysis (SPA) to detect LSB steganography.
    
    Args:
        pixels: Numpy array of pixel values
    
    Returns:
        Likelihood based on SPA (0-1)
    """
    likelihoods = []
    
    for channel in range(3):
        # Get channel data
        channel_data = pixels[:, :, channel]
        height, width = channel_data.shape
        
        # Count pairs
        regular_pairs = 0
        singular_pairs = 0
        
        # Sample a subset of pixels for better performance
        num_samples = min(10000, height * width // 2)
        samples = []
        
        for _ in range(num_samples):
            # Pick a random pixel and its neighbor
            y = random.randint(0, height - 2)
            x = random.randint(0, width - 1)
            
            # Get values of the pixel and its neighbor
            value1 = int(channel_data[y, x])
            value2 = int(channel_data[y + 1, x])
            
            samples.append((value1, value2))
        
        # Analyze collected samples
        for value1, value2 in samples:
            # Check if pair is regular or singular
            # Regular pair: values differ by 0 or 1 in LSB
            if (value1 // 2 == value2 // 2):
                regular_pairs += 1
            # Singular pair: values differ by 2k+1
            elif abs(value1 - value2) % 2 == 1:
                singular_pairs += 1
        
        # Calculate ratio of regular to singular pairs
        # In a clean image, this ratio should be close to 1
        # In a stego image, it often deviates
        if singular_pairs > 0:
            ratio = regular_pairs / singular_pairs
            
            # Calculate how much the ratio deviates from the expected value of 1
            deviation = abs(ratio - 1)
            
            # Convert deviation to likelihood
            # Higher deviation means higher likelihood of steganography
            channel_likelihood = min(deviation / 0.5, 1.0)
            likelihoods.append(channel_likelihood)
        else:
            likelihoods.append(0.5)  # Neutral if no singular pairs
    
    # Take the maximum likelihood across channels
    spa_likelihood = max(likelihoods)
    
    # Scale the result
    spa_likelihood = scale_likelihood(spa_likelihood)
    
    return spa_likelihood

def analyze_rgb_correlation(pixels):
    """
    Analyze correlation between RGB channels for signs of steganography.
    
    Args:
        pixels: Numpy array of pixel values
    
    Returns:
        Likelihood based on RGB correlation analysis (0-1)
    """
    # In natural images, RGB channels are usually correlated
    # Steganography can disrupt this correlation
    
    # Sample a subset of pixels for performance
    height, width, _ = pixels.shape
    num_samples = min(50000, height * width)
    
    # Randomly select indices
    indices = np.random.choice(height * width, num_samples, replace=False)
    y_indices = indices // width
    x_indices = indices % width
    
    # Extract sampled pixels
    r_samples = pixels[y_indices, x_indices, 0].flatten()
    g_samples = pixels[y_indices, x_indices, 1].flatten()
    b_samples = pixels[y_indices, x_indices, 2].flatten()
    
    # Calculate correlation coefficients
    rg_corr = np.corrcoef(r_samples, g_samples)[0, 1]
    rb_corr = np.corrcoef(r_samples, b_samples)[0, 1]
    gb_corr = np.corrcoef(g_samples, b_samples)[0, 1]
    
    # Handle NaN values
    if np.isnan(rg_corr):
        rg_corr = 0
    if np.isnan(rb_corr):
        rb_corr = 0
    if np.isnan(gb_corr):
        gb_corr = 0
    
    # Calculate average correlation
    avg_corr = (abs(rg_corr) + abs(rb_corr) + abs(gb_corr)) / 3
    
    # In natural images, correlation is typically high (0.7-0.95)
    # Lower correlation may indicate steganography
    
    # Convert to likelihood (lower correlation = higher likelihood)
    # Typically, correlation below 0.5 is unusual for natural images
    likelihood = 1 - avg_corr
    
    # Scale based on typical values
    # A correlation of 0.7 or higher is typical (likelihood <= 0.3)
    # A correlation of 0.5 is somewhat suspicious (likelihood ~ 0.5)
    # A correlation of 0.3 or lower is highly suspicious (likelihood >= 0.7)
    likelihood = scale_likelihood(likelihood, center=0.5, sensitivity=2.0)
    
    return likelihood

# Utility functions
def count_runs(binary_data):
    """Count the number of runs in binary data."""
    runs = 0
    for i in range(1, len(binary_data)):
        if binary_data[i] != binary_data[i-1]:
            runs += 1
    return runs

def calculate_entropy(data):
    """Calculate Shannon entropy."""
    if len(data) <= 1:
        return 0
    
    # Count occurrences of each unique value
    values, counts = np.unique(data, return_counts=True)
    probabilities = counts / len(data)
    
    # Calculate entropy
    entropy = -np.sum(probabilities * np.log2(probabilities))
    return entropy

def scale_likelihood(value, center=0.5, sensitivity=1.0):
    """
    Scale a likelihood value to make results more decisive.
    
    Args:
        value: Original likelihood value (0-1)
        center: Center point for the scaling curve (0-1)
        sensitivity: Controls how aggressively to scale (higher = more aggressive)
    
    Returns:
        Scaled likelihood value (0-1)
    """
    # Increase the value first to make detection more aggressive
    # This makes the tool more likely to report potential steganography
    boosted_value = value * 1.7
    
    # Apply sigmoid-like scaling centered at the specified point with higher sensitivity
    scaled = 1 / (1 + np.exp(-sensitivity * 2.0 * (boosted_value - center) * 10))
    
    # Ensure the result is in the range [0, 1]
    return max(0, min(1, scaled))

def determine_potential_techniques(result):
    """
    Determine potential steganography techniques based on detection results.
    
    Args:
        result: DetectionResult object
    
    Returns:
        List of potential techniques
    """
    techniques = []
    indicators = result.indicators
    
    # LSB Steganography
    if "LSB Analysis" in indicators and indicators["LSB Analysis"]["value"] > 0.6:
        techniques.append("LSB Steganography")
    
    # Metadata embedding
    if "Metadata Analysis" in indicators and indicators["Metadata Analysis"]["value"] > 0.6:
        techniques.append("Metadata Embedding")
    
    # Frequency domain techniques (DCT, etc.)
    if ("Noise Analysis" in indicators and indicators["Noise Analysis"]["value"] > 0.7 and
            "Histogram Analysis" in indicators and indicators["Histogram Analysis"]["value"] > 0.5):
        techniques.append("Frequency Domain Steganography (DCT)")
    
    # Sample pair analysis suggests LSB replacement
    if "Sample Pair Analysis" in indicators and indicators["Sample Pair Analysis"]["value"] > 0.7:
        if "LSB Steganography" not in techniques:
            techniques.append("LSB Replacement")
    
    # If high overall likelihood but no specific technique identified
    if result.likelihood > 0.7 and not techniques:
        techniques.append("Unknown Steganographic Technique")
    
    return techniques