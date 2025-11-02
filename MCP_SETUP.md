# DEEP ANAL MCP Server Setup

This MCP server exposes your steganography analysis tools to Claude and other MCP clients.

## What This Enables

Claude can now:
- Analyze images for hidden steganographic content
- Perform quick scans for suspicious patterns
- Extract hidden data from images
- Calculate entropy and analyze file structure
- Extract readable strings from files

## Setup for Claude Desktop

### 1. Find Your Config File

**macOS:**
```
~/Library/Application Support/Claude/claude_desktop_config.json
```

**Windows:**
```
%APPDATA%\Claude\claude_desktop_config.json
```

**Linux:**
```
~/.config/Claude/claude_desktop_config.json
```

### 2. Add DEEP ANAL MCP Server

Edit the config file and add:

```json
{
  "mcpServers": {
    "deep-anal": {
      "command": "python",
      "args": ["/path/to/your/replit/workspace/mcp_server.py"],
      "env": {
        "OPENAI_API_KEY": "your-openai-key-if-needed"
      }
    }
  }
}
```

Replace `/path/to/your/replit/workspace/` with the actual path to this Replit project.

### 3. Restart Claude Desktop

Close and reopen Claude Desktop to load the MCP server.

## Available Tools

Once connected, Claude can use these tools:

### 1. `analyze_image_steganography`
Full comprehensive steganography analysis
- **Input:** `image_path` (string)
- **Returns:** Detailed analysis with likelihood, indicators, metadata, entropy

### 2. `quick_scan_image`
Fast detection scan
- **Input:** `image_path` (string)
- **Returns:** Quick likelihood score and key indicators

### 3. `extract_hidden_content`
Extract hidden data using multiple techniques
- **Input:** `image_path` (string), `max_attempts` (int, optional)
- **Returns:** Extracted content from successful methods

### 4. `get_file_entropy_analysis`
Calculate and interpret file entropy
- **Input:** `image_path` (string)
- **Returns:** Entropy score with interpretation

### 5. `extract_strings_from_file`
Extract readable strings
- **Input:** `file_path` (string), `min_length` (int, optional)
- **Returns:** Strings, URLs, emails, suspicious keywords

## Usage Examples

Once set up, you can ask Claude:

```
"Analyze this image for steganography: /path/to/image.png"
"Quick scan this file: /path/to/suspicious.jpg"
"Extract any hidden content from: /path/to/test.png"
"What's the entropy of this file: /path/to/data.bin"
```

Claude will automatically use the appropriate MCP tools to analyze the files.

## Testing

Test the server manually:

```bash
# Make the server executable
chmod +x mcp_server.py

# Run the server (it will start in MCP stdio mode)
python mcp_server.py
```

## Troubleshooting

**Server not appearing in Claude:**
- Check the config file path is correct
- Verify Python path in the command
- Ensure all dependencies are installed
- Check Claude Desktop logs for errors

**Tools not working:**
- Verify the image paths are absolute paths
- Ensure the MCP server has read permissions
- Check that required system tools (exiftool, etc.) are installed

## Resources

- Tool help: Available via the `analysis://help` resource in Claude
- MCP Documentation: https://modelcontextprotocol.io
- DEEP ANAL Docs: See README.md and replit.md
