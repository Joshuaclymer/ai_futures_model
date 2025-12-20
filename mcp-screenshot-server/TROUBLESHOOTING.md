# MCP Screenshot Server Troubleshooting

## Overview
This MCP server uses Puppeteer to take screenshots of web pages. It's configured in `.mcp.json` files at project roots.

## Common Issues

### 1. Navigation Timeout on Localhost
**Symptom:** `Navigation timeout of 30000 ms exceeded` when screenshotting localhost URLs.

**Cause:** The original code used `waitUntil: "networkidle2"` which waits for all network activity to stop. Next.js and other dev servers keep WebSocket connections open for hot module reloading, so `networkidle2` never resolves.

**Fix (already applied 2024-12):** Modified `index.js` to use `waitUntil: "load"` for localhost/127.0.0.1 URLs:
```javascript
const isLocalhost = url.includes('localhost') || url.includes('127.0.0.1') || url.includes('10.0.0.');
await page.goto(url, {
  waitUntil: isLocalhost ? "load" : "networkidle2",
  timeout: 30000,
});
```

### 2. "Not connected" Error
**Symptom:** MCP tool returns "Not connected" error.

**Cause:** The MCP server process was killed or crashed and Claude Code hasn't reconnected.

**Fix:** Restart Claude Code. The MCP server auto-starts based on `.mcp.json` config.

### 3. MCP Server Not Available in Project
**Symptom:** Screenshot tool not listed in available MCP tools.

**Cause:** The project doesn't have `.mcp.json` configured.

**Fix:** Copy or create `.mcp.json` at project root:
```json
{
  "mcpServers": {
    "screenshot": {
      "command": "node",
      "args": ["/Users/joshuaclymer/github/ai_futures_simulator/mcp-screenshot-server/index.js"]
    }
  }
}
```

## Restarting the MCP Server

1. Find the process: `ps aux | grep mcp-screenshot`
2. Kill it: `kill <PID>`
3. Restart Claude Code (the server auto-starts via stdio transport)

## Key Files
- `index.js` - Main server code
- `package.json` - Dependencies (puppeteer, @modelcontextprotocol/sdk)
- Project `.mcp.json` files - Configure which projects can use this server

## Testing
After changes, test with:
1. External URL: `https://example.com` (should always work)
2. Localhost: `http://localhost:3000` (needs the networkidle2 fix)
