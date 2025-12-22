# Viewing Frontend Applications

This guide explains how to start the frontend servers and take screenshots for development and debugging.

## Frontend Servers

### 1. App Frontend (Next.js) - Port 3000

The main AI Futures Simulator frontend runs on Next.js.

**Start the server:**
```bash
cd /Users/joshuaclymer/github/ai_futures_simulator/app_frontend
npm run next-dev
```

**View in browser:** http://localhost:3000

**Key pages:**
- `/ai-black-projects` - Dark compute model visualization

### 2. Black Project Frontend (Flask) - Port 5001

The Black Project frontend is served by Flask using Jinja templates.

**Start the server:**
```bash
cd /Users/joshuaclymer/github/covert_compute_production_model
source venv/bin/activate
python3 app.py
```

**View in browser:** http://localhost:5001

## Taking Screenshots with Claude Code

Claude Code has access to an MCP screenshot tool that can capture web pages programmatically.

### Basic Screenshot

Ask Claude to take a screenshot:
```
Take a screenshot of http://localhost:5001/
```

### Screenshot Options

The screenshot tool supports these parameters:
- `url` - The URL to capture (required)
- `width` - Viewport width in pixels (default: 1280)
- `height` - Viewport height in pixels (default: 800)
- `fullPage` - Capture entire scrollable page (default: false)
- `waitTime` - Milliseconds to wait before capture (default: 1000)
- `waitForSelector` - CSS selector to wait for before capturing

### Examples

**Standard screenshot:**
```
Take a screenshot of http://localhost:3000/ai-black-projects with width 1280 and height 900
```

**Full page capture:**
```
Take a full page screenshot of http://localhost:5001/
```

**Wait for content to load:**
```
Take a screenshot of http://localhost:5001/ and wait 3000ms for content to load
```

## Troubleshooting

### Navigation Timeout on Localhost

If screenshots timeout on localhost URLs, the Next.js dev server may be overloaded or stuck.

**Fix:** Restart the server:
```bash
# Find the process
lsof -i :3000 -sTCP:LISTEN

# Kill it
kill <PID>

# Restart
cd /Users/joshuaclymer/github/ai_futures_simulator/app_frontend && npm run next-dev
```

### Flask Module Not Found

If Flask isn't found, activate the virtual environment first:
```bash
cd /Users/joshuaclymer/github/covert_compute_production_model
source venv/bin/activate
python3 app.py
```

### MCP Tool Not Available

If the screenshot tool isn't available, check that the MCP server is configured in `.mcp.json` at the project root.

See `/Users/joshuaclymer/github/ai_futures_simulator/mcp-screenshot-server/TROUBLESHOOTING.md` for detailed MCP troubleshooting.

## Quick Reference

| Frontend | Port | Start Command |
|----------|------|---------------|
| Next.js App | 3000 | `npm run next-dev` (from app_frontend) |
| Black Project | 5001 | `python3 app.py` (with venv activated) |
