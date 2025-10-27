"""
PGP/GPG Analysis Module for Steganography Detection
Detects and analyzes PGP encrypted messages, keys, and signatures in extracted content.
"""

import re
import base64
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

@dataclass
class PGPBlock:
    """Represents a detected PGP/GPG block."""
    block_type: str  # MESSAGE, PUBLIC KEY, PRIVATE KEY, SIGNATURE, etc.
    content: str
    header: str
    footer: str
    version: Optional[str] = None
    charset: Optional[str] = None
    key_id: Optional[str] = None
    checksum: Optional[str] = None
    size_bytes: int = 0


class PGPAnalyzer:
    """Analyzer for PGP/GPG encrypted content and keys."""
    
    # PGP armor block patterns
    PGP_PATTERNS = {
        'message': (
            r'-----BEGIN PGP MESSAGE-----\s*(.*?)\s*-----END PGP MESSAGE-----',
            'Encrypted Message'
        ),
        'signed_message': (
            r'-----BEGIN PGP SIGNED MESSAGE-----\s*(.*?)\s*-----END PGP SIGNATURE-----',
            'Signed Message'
        ),
        'public_key': (
            r'-----BEGIN PGP PUBLIC KEY BLOCK-----\s*(.*?)\s*-----END PGP PUBLIC KEY BLOCK-----',
            'Public Key'
        ),
        'private_key': (
            r'-----BEGIN PGP PRIVATE KEY BLOCK-----\s*(.*?)\s*-----END PGP PRIVATE KEY BLOCK-----',
            'Private Key'
        ),
        'signature': (
            r'-----BEGIN PGP SIGNATURE-----\s*(.*?)\s*-----END PGP SIGNATURE-----',
            'Signature'
        ),
    }
    
    def __init__(self):
        self.detected_blocks: List[PGPBlock] = []
        
    def analyze_content(self, content: str) -> Dict:
        """
        Analyze content for PGP/GPG blocks.
        
        Args:
            content: Text content to analyze
            
        Returns:
            Dictionary with analysis results
        """
        if not content or not isinstance(content, str):
            return {
                'has_pgp': False,
                'blocks': [],
                'summary': 'No content to analyze'
            }
        
        self.detected_blocks = []
        
        # Detect all PGP blocks
        for block_type, (pattern, description) in self.PGP_PATTERNS.items():
            matches = re.finditer(pattern, content, re.DOTALL | re.MULTILINE)
            
            for match in matches:
                block = self._parse_pgp_block(
                    match.group(0),
                    match.group(1),
                    block_type,
                    description
                )
                if block:
                    self.detected_blocks.append(block)
        
        # Generate analysis report
        return self._generate_report()
    
    def _parse_pgp_block(self, full_text: str, body: str, 
                         block_type: str, description: str) -> Optional[PGPBlock]:
        """Parse a PGP armor block and extract metadata."""
        try:
            lines = full_text.split('\n')
            header = lines[0] if lines else ''
            footer = lines[-1] if len(lines) > 1 else ''
            
            # Extract headers (Version, Charset, etc.)
            version = None
            charset = None
            
            for line in lines[1:]:
                if line.startswith('Version:'):
                    version = line.split(':', 1)[1].strip()
                elif line.startswith('Charset:'):
                    charset = line.split(':', 1)[1].strip()
                elif line.strip() == '':
                    break  # Empty line marks end of headers
            
            # Try to extract key ID from content
            key_id = self._extract_key_id(body)
            
            # Calculate approximate size
            # Remove headers and whitespace for size calculation
            clean_body = re.sub(r'[^A-Za-z0-9+/=]', '', body)
            size_bytes = len(clean_body) * 3 // 4  # Approximate base64 decoded size
            
            # Extract checksum if present
            checksum = self._extract_checksum(body)
            
            return PGPBlock(
                block_type=description,
                content=body.strip(),
                header=header,
                footer=footer,
                version=version,
                charset=charset,
                key_id=key_id,
                checksum=checksum,
                size_bytes=size_bytes
            )
            
        except Exception as e:
            return None
    
    def _extract_key_id(self, content: str) -> Optional[str]:
        """Attempt to extract key ID from PGP content."""
        try:
            # Remove headers and whitespace
            clean_content = re.sub(r'[^A-Za-z0-9+/=]', '', content)
            
            # Try to decode base64
            decoded = base64.b64decode(clean_content[:100])  # First 100 chars
            
            # Look for key ID patterns in decoded data
            # Key IDs are typically 8 or 16 hex characters
            hex_str = decoded.hex().upper()
            
            # Common key ID positions in PGP packets
            # This is a simplified heuristic
            if len(hex_str) >= 16:
                # Try to find patterns that look like key IDs
                return hex_str[8:24] if len(hex_str) >= 24 else hex_str[:16]
                
        except Exception:
            pass
        
        return None
    
    def _extract_checksum(self, content: str) -> Optional[str]:
        """Extract CRC24 checksum if present."""
        # PGP uses CRC24 checksum at the end, starts with =
        checksum_pattern = r'=([A-Za-z0-9+/]{4})'
        match = re.search(checksum_pattern, content)
        
        if match:
            return match.group(1)
        
        return None
    
    def _generate_report(self) -> Dict:
        """Generate analysis report from detected blocks."""
        if not self.detected_blocks:
            return {
                'has_pgp': False,
                'blocks': [],
                'summary': 'No PGP/GPG blocks detected',
                'risk_level': 'low',
                'indicators': []
            }
        
        # Categorize blocks
        block_types = {}
        for block in self.detected_blocks:
            block_type = block.block_type
            if block_type not in block_types:
                block_types[block_type] = []
            block_types[block_type].append(block)
        
        # Assess risk level
        risk_level = self._assess_risk(block_types)
        
        # Generate indicators
        indicators = self._generate_indicators(block_types)
        
        # Create summary
        summary = self._create_summary(block_types)
        
        # Format blocks for output
        formatted_blocks = []
        for block in self.detected_blocks:
            formatted_blocks.append({
                'type': block.block_type,
                'version': block.version,
                'charset': block.charset,
                'key_id': block.key_id,
                'size_bytes': block.size_bytes,
                'checksum': block.checksum,
                'content_preview': block.content[:100] + '...' if len(block.content) > 100 else block.content
            })
        
        return {
            'has_pgp': True,
            'blocks': formatted_blocks,
            'block_count': len(self.detected_blocks),
            'block_types': {k: len(v) for k, v in block_types.items()},
            'summary': summary,
            'risk_level': risk_level,
            'indicators': indicators,
            'recommendations': self._generate_recommendations(block_types)
        }
    
    def _assess_risk(self, block_types: Dict) -> str:
        """Assess security risk level based on detected blocks."""
        # Private keys are high risk
        if 'Private Key' in block_types:
            return 'critical'
        
        # Encrypted messages with no key context
        if 'Encrypted Message' in block_types:
            return 'high'
        
        # Public keys or signatures
        if 'Public Key' in block_types or 'Signature' in block_types:
            return 'medium'
        
        return 'low'
    
    def _generate_indicators(self, block_types: Dict) -> List[str]:
        """Generate security indicators based on detected blocks."""
        indicators = []
        
        if 'Private Key' in block_types:
            indicators.append('🔴 CRITICAL: Private key detected - potential key compromise')
        
        if 'Encrypted Message' in block_types:
            indicators.append('🟡 Encrypted message found - hidden communication')
        
        if 'Public Key' in block_types:
            indicators.append('🟢 Public key detected - used for encryption/verification')
        
        if 'Signature' in block_types:
            indicators.append('🟢 Digital signature found - authenticity verification')
        
        if 'Signed Message' in block_types:
            indicators.append('🟡 Signed message detected - verify sender identity')
        
        # Check for multiple block types
        if len(block_types) > 2:
            indicators.append('⚠️  Multiple PGP block types - complex cryptographic workflow')
        
        return indicators
    
    def _create_summary(self, block_types: Dict) -> str:
        """Create human-readable summary."""
        total_blocks = sum(len(blocks) for blocks in block_types.values())
        
        type_list = ', '.join([f"{count} {btype}(s)" 
                              for btype, blocks in block_types.items() 
                              for count in [len(blocks)]])
        
        return f"Detected {total_blocks} PGP/GPG block(s): {type_list}"
    
    def _generate_recommendations(self, block_types: Dict) -> List[str]:
        """Generate investigation recommendations."""
        recommendations = []
        
        if 'Private Key' in block_types:
            recommendations.append('URGENT: Investigate private key exposure - potential security breach')
            recommendations.append('Determine if key is encrypted with passphrase')
            recommendations.append('Check if key has been revoked or compromised')
        
        if 'Encrypted Message' in block_types:
            recommendations.append('Attempt to decrypt message with available keys')
            recommendations.append('Analyze key ID to identify sender/recipient')
            recommendations.append('Check for related public keys in the same file')
        
        if 'Public Key' in block_types:
            recommendations.append('Extract key ID and search key servers')
            recommendations.append('Verify key fingerprint for authenticity')
            recommendations.append('Check key creation date and expiration')
        
        if 'Signature' in block_types:
            recommendations.append('Verify signature against public key')
            recommendations.append('Check signature timestamp for validity')
            recommendations.append('Confirm signer identity')
        
        # General recommendations
        recommendations.append('Document all PGP blocks for chain of custody')
        recommendations.append('Consider running gpg --list-packets for detailed analysis')
        
        return recommendations


def detect_pgp_in_text(text: str) -> bool:
    """
    Quick check if text contains PGP armor blocks.
    
    Args:
        text: Text to check
        
    Returns:
        True if PGP blocks detected
    """
    if not text or not isinstance(text, str):
        return False
    
    pgp_markers = [
        '-----BEGIN PGP',
        '-----END PGP'
    ]
    
    return any(marker in text for marker in pgp_markers)


def analyze_pgp_content(content: str) -> Dict:
    """
    Convenience function to analyze PGP content.
    
    Args:
        content: Text content to analyze
        
    Returns:
        Analysis results dictionary
    """
    analyzer = PGPAnalyzer()
    return analyzer.analyze_content(content)
