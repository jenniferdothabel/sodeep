"""
AI Assistant for Steganography Analysis
Provides intelligent analysis and investigation guidance using OpenAI's GPT models.
"""

import os
import json
from openai import OpenAI

# the newest OpenAI model is "gpt-5" which was released August 7, 2025.
# do not change this unless explicitly requested by the user
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

class SteganographyAssistant:
    """AI assistant for steganography analysis and investigation guidance."""
    
    def __init__(self):
        self.model = "gpt-5"  # Using the latest model
        
    def analyze_detection_results(self, detection_result, file_metadata, extracted_content=None):
        """
        Analyze steganography detection results and provide expert insights.
        
        Args:
            detection_result: DetectionResult object from stego_detector
            file_metadata: File metadata dictionary
            extracted_content: Any extracted hidden content
            
        Returns:
            dict with analysis, recommendations, and next steps
        """
        try:
            # Prepare context for the AI
            context = {
                "likelihood": detection_result.likelihood if hasattr(detection_result, 'likelihood') else 0,
                "indicators": detection_result.indicators if hasattr(detection_result, 'indicators') else {},
                "explanation": detection_result.explanation if hasattr(detection_result, 'explanation') else "",
                "metadata": file_metadata,
                "has_extracted_content": extracted_content is not None,
                "extracted_preview": str(extracted_content)[:200] if extracted_content else None
            }
            
            prompt = f"""You are an expert digital forensics investigator specializing in steganography analysis. 
            
Analyze the following steganography detection results and provide professional insights:

DETECTION RESULTS:
- Overall Likelihood: {context['likelihood']:.1%}
- Technical Explanation: {context['explanation']}
- Individual Indicators: {json.dumps(context['indicators'], indent=2)}

FILE METADATA:
{json.dumps(context['metadata'], indent=2)}

EXTRACTED CONTENT:
- Content Found: {context['has_extracted_content']}
- Preview: {context['extracted_preview'] or 'None'}

Please provide a professional analysis in JSON format with these sections:
1. "summary": Brief assessment of the likelihood and significance
2. "technical_analysis": Detailed explanation of what the indicators mean
3. "investigation_recommendations": Specific next steps for the investigation
4. "potential_techniques": Likely steganography methods used
5. "risk_assessment": Security implications if this is malicious
6. "plain_language": Simple explanation for non-technical users

Be concise but thorough. Focus on actionable insights."""

            response = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"}
            )
            
            analysis = json.loads(response.choices[0].message.content)
            return analysis
            
        except Exception as e:
            return {
                "summary": f"AI analysis unavailable: {str(e)}",
                "technical_analysis": "Unable to perform automated analysis at this time.",
                "investigation_recommendations": ["Manual review of detection results", "Try different extraction methods"],
                "potential_techniques": ["Unknown"],
                "risk_assessment": "Manual assessment required",
                "plain_language": "The AI assistant couldn't analyze this image right now. Please review the detection results manually."
            }
    
    def interpret_extracted_content(self, extracted_data, extraction_method):
        """
        Analyze extracted content and provide interpretation.
        
        Args:
            extracted_data: The extracted hidden content
            extraction_method: Method used for extraction
            
        Returns:
            dict with content analysis and recommendations
        """
        try:
            # Prepare content for analysis
            if isinstance(extracted_data, bytes):
                # Try to decode as text
                try:
                    text_content = extracted_data.decode('utf-8', errors='ignore')
                    content_type = "text"
                    content_preview = text_content[:500]
                except:
                    content_type = "binary"
                    content_preview = f"Binary data: {len(extracted_data)} bytes"
            else:
                content_type = "text"
                content_preview = str(extracted_data)[:500]
            
            prompt = f"""You are a digital forensics expert analyzing extracted hidden content from steganography.

EXTRACTION DETAILS:
- Method Used: {extraction_method}
- Content Type: {content_type}
- Content Preview: {content_preview}

Analyze this extracted content and provide insights in JSON format:
1. "content_type_analysis": What type of data this appears to be
2. "potential_purpose": Why this might have been hidden
3. "security_implications": Any security concerns
4. "decoding_suggestions": If this looks encoded/encrypted, suggest decoding methods
5. "investigation_priority": How urgent is further investigation (low/medium/high)
6. "next_steps": Specific actions to take with this content

Be professional and focus on forensic analysis."""

            response = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"}
            )
            
            return json.loads(response.choices[0].message.content)
            
        except Exception as e:
            return {
                "content_type_analysis": f"Analysis failed: {str(e)}",
                "potential_purpose": "Unknown",
                "security_implications": "Manual review required",
                "decoding_suggestions": ["Manual analysis recommended"],
                "investigation_priority": "medium",
                "next_steps": ["Save content for manual analysis", "Try different extraction methods"]
            }
    
    def generate_investigation_report(self, filename, detection_results, extracted_contents, metadata):
        """
        Generate a comprehensive investigation report.
        
        Args:
            filename: Name of the analyzed file
            detection_results: Detection analysis results
            extracted_contents: List of extracted content results
            metadata: File metadata
            
        Returns:
            Formatted investigation report as string
        """
        try:
            context = {
                "filename": filename,
                "detection": detection_results,
                "extractions": len(extracted_contents) if extracted_contents else 0,
                "metadata_keys": list(metadata.keys()) if metadata else []
            }
            
            prompt = f"""Generate a professional digital forensics investigation report for the following steganography analysis:

FILE: {context['filename']}
DETECTION LIKELIHOOD: {detection_results.get('likelihood', 0) if isinstance(detection_results, dict) else 'Unknown'}
EXTRACTED ITEMS: {context['extractions']}
METADATA FIELDS: {context['metadata_keys']}

Create a structured report with:
1. Executive Summary
2. Technical Findings
3. Evidence Summary
4. Recommendations
5. Conclusion

Use professional forensics language but keep it accessible. Focus on facts and actionable insights."""

            response = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}]
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            return f"""
STEGANOGRAPHY ANALYSIS REPORT
============================

File: {filename}
Analysis Date: {context.get('timestamp', 'Unknown')}

EXECUTIVE SUMMARY:
AI report generation encountered an error: {str(e)}

TECHNICAL FINDINGS:
Manual review of detection results recommended.

RECOMMENDATIONS:
1. Review detection indicators manually
2. Verify extraction results
3. Consider additional analysis methods

CONCLUSION:
Further investigation required due to analysis limitations.
"""

def get_investigation_suggestions(likelihood, indicators):
    """
    Get quick investigation suggestions based on detection results.
    
    Args:
        likelihood: Detection likelihood (0-1)
        indicators: Detection indicators dictionary
        
    Returns:
        List of suggested next steps
    """
    suggestions = []
    
    if likelihood >= 0.7:
        suggestions.extend([
            "🔴 High likelihood detected - immediate extraction recommended",
            "Try multiple extraction methods to find hidden content",
            "Check for password-protected steganography",
            "Analyze metadata for additional clues"
        ])
    elif likelihood >= 0.4:
        suggestions.extend([
            "🟡 Moderate likelihood - worth investigating further",
            "Try LSB extraction on different color channels", 
            "Check for common steganography tools signatures",
            "Compare with known clean versions if available"
        ])
    else:
        suggestions.extend([
            "🟢 Low likelihood - appears normal",
            "Consider if this might be a red herring",
            "Check for advanced or custom steganography methods",
            "Verify file integrity and authenticity"
        ])
    
    # Add specific suggestions based on indicators
    if indicators:
        if any('lsb' in name.lower() for name in indicators.keys()):
            suggestions.append("LSB patterns detected - focus on bit plane analysis")
        if any('metadata' in name.lower() for name in indicators.keys()):
            suggestions.append("Metadata anomalies found - examine EXIF data closely")
        if any('histogram' in name.lower() for name in indicators.keys()):
            suggestions.append("Histogram irregularities - check for frequency domain hiding")
    
    return suggestions[:6]  # Limit to top 6 suggestions