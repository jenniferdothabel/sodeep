# ChatGPT Action Setup for DEEP ANAL Steganography Analyzer

## Quick Setup Instructions

### 1. Deploy Your API
Your steganography analysis API is available at:
```
https://your-replit-url.replit.app/
```

### 2. Create ChatGPT Action

1. Go to ChatGPT and click your profile
2. Select "My GPTs" 
3. Click "Create a GPT"
4. Configure your action:

**Name:** DEEP ANAL Steganography Analyzer
**Description:** Advanced steganography analysis tool that can detect and extract hidden data from images

### 3. Action Configuration

In the "Configure" tab, add this OpenAPI schema:

```json
{
  "openapi": "3.0.0",
  "info": {
    "title": "DEEP ANAL Steganography API",
    "description": "Advanced steganography analysis and hidden content extraction",
    "version": "1.0.0"
  },
  "servers": [
    {
      "url": "https://your-replit-url.replit.app",
      "description": "DEEP ANAL Analysis Server"
    }
  ],
  "paths": {
    "/api/analyze": {
      "post": {
        "summary": "Analyze image for steganography",
        "description": "Performs comprehensive steganography analysis including detection, extraction, and AI insights",
        "operationId": "analyzeImage",
        "requestBody": {
          "required": true,
          "content": {
            "multipart/form-data": {
              "schema": {
                "type": "object",
                "properties": {
                  "file": {
                    "type": "string",
                    "format": "binary",
                    "description": "Image file to analyze (PNG or JPEG)"
                  }
                },
                "required": ["file"]
              }
            }
          }
        },
        "responses": {
          "200": {
            "description": "Analysis completed successfully",
            "content": {
              "application/json": {
                "schema": {
                  "type": "object",
                  "properties": {
                    "analysis_summary": {
                      "type": "object",
                      "description": "Summary of analysis results including likelihood percentage"
                    },
                    "detection_details": {
                      "type": "object",
                      "description": "Detailed steganography detection indicators"
                    },
                    "extracted_content": {
                      "type": "array",
                      "description": "Any hidden content found and extracted"
                    },
                    "ai_insights": {
                      "type": "object",
                      "description": "AI-powered analysis and investigation recommendations"
                    }
                  }
                }
              }
            }
          }
        }
      }
    },
    "/api/quick-scan": {
      "post": {
        "summary": "Quick steganography scan",
        "description": "Fast detection scan to check likelihood of hidden data",
        "operationId": "quickScan",
        "requestBody": {
          "required": true,
          "content": {
            "multipart/form-data": {
              "schema": {
                "type": "object",
                "properties": {
                  "file": {
                    "type": "string",
                    "format": "binary",
                    "description": "Image file to scan"
                  }
                },
                "required": ["file"]
              }
            }
          }
        },
        "responses": {
          "200": {
            "description": "Quick scan completed",
            "content": {
              "application/json": {
                "schema": {
                  "type": "object",
                  "properties": {
                    "steganography_likelihood": {
                      "type": "number",
                      "description": "Probability of hidden content (0-1)"
                    },
                    "risk_assessment": {
                      "type": "string",
                      "description": "Risk level: low, medium, or high"
                    },
                    "recommended_action": {
                      "type": "string",
                      "description": "Suggested next step for investigation"
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }
}
```

### 4. Instructions for ChatGPT

Add these instructions in the "Instructions" section:

```
You are DEEP ANAL, an expert steganography analysis assistant. You help users detect and extract hidden data from images using advanced digital forensics techniques.

When a user uploads an image:

1. **First, run a quick scan** using the quick-scan endpoint to give an immediate assessment
2. **If likelihood is moderate to high (>30%)**, automatically run the full analysis
3. **Interpret results in plain language** - explain what steganography likelihood means
4. **Provide actionable recommendations** based on the risk level
5. **If hidden content is found**, explain what it might be and suggest next steps

**Response Format:**
- Start with a clear assessment: "🔍 **Steganography Analysis Complete**"
- Show the likelihood percentage with appropriate emoji (🟢<30%, 🟡30-70%, 🔴>70%)
- Explain findings in non-technical terms
- List any extracted content clearly
- Provide specific recommendations for investigation

**Key Capabilities to Mention:**
- LSB (Least Significant Bit) detection
- Metadata analysis
- Multiple extraction methods
- AI-powered insights
- Risk assessment

**Security Focused:**
- Always warn if high-risk content is detected
- Suggest appropriate security measures
- Recommend further investigation when needed
- Explain potential implications of findings

Be thorough but accessible. Users may not be security experts.
```

### 5. Test Your Action

Once configured, test with sample images:

1. **Normal image** - should show low likelihood
2. **Image with hidden text** - should detect and extract content
3. **Password-protected steganography** - should identify but not extract

### 6. Usage Examples

**ChatGPT User:** "Can you check this image for hidden data?"

**Expected Response:**
```
🔍 **Steganography Analysis Complete**

🟡 **Detection Likelihood: 67.3%** - Moderate to high probability of hidden content

**Analysis Summary:**
- File: suspicious_image.png (245 KB)
- Multiple indicators suggest LSB steganography
- Entropy levels higher than normal
- Histogram anomalies detected

**🔓 Hidden Content Found:**
- Method: LSB (Channel: 0, Bit: 0)
- Extracted Text: "Secret message: Meeting at midnight"
- Confidence: 89%

**🎯 Recommendations:**
1. Investigate the source of this image
2. Check for additional hidden layers
3. Analyze metadata for more clues
4. Consider this medium-risk content

**Next Steps:**
The extracted message appears to be plaintext. Consider the context and source of this image for security implications.
```

## API Endpoints Reference

### POST /api/analyze
Full steganography analysis with extraction and AI insights

### POST /api/quick-scan  
Fast detection scan for initial assessment

### GET /api/health
Health check endpoint

## Authentication
Currently no authentication required. Consider adding API keys for production use.

## Rate Limits
No current rate limits. Monitor usage and implement as needed.

## Error Handling
API returns appropriate HTTP status codes and error messages in JSON format.