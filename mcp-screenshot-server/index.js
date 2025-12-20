#!/usr/bin/env node

import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import {
  CallToolRequestSchema,
  ListToolsRequestSchema,
} from "@modelcontextprotocol/sdk/types.js";
import puppeteer from "puppeteer";

// Maximum height to avoid exceeding Claude's image dimension limits (8192x8192 max, using 6000 for margin)
const MAX_SCREENSHOT_HEIGHT = 6000;

const server = new Server(
  {
    name: "screenshot-server",
    version: "1.0.0",
  },
  {
    capabilities: {
      tools: {},
    },
  }
);

// List available tools
server.setRequestHandler(ListToolsRequestSchema, async () => {
  return {
    tools: [
      {
        name: "screenshot",
        description:
          "Take a screenshot of a web page and return it as a base64-encoded image. Use this to view web pages in a browser.",
        inputSchema: {
          type: "object",
          properties: {
            url: {
              type: "string",
              description: "The URL of the web page to screenshot",
            },
            fullPage: {
              type: "boolean",
              description:
                "Whether to capture the full scrollable page (default: false)",
              default: false,
            },
            width: {
              type: "number",
              description: "Viewport width in pixels (default: 1280)",
              default: 1280,
            },
            height: {
              type: "number",
              description: "Viewport height in pixels (default: 800)",
              default: 800,
            },
            waitForSelector: {
              type: "string",
              description:
                "Optional CSS selector to wait for before taking the screenshot",
            },
            waitTime: {
              type: "number",
              description:
                "Optional time in milliseconds to wait before taking the screenshot (default: 1000)",
              default: 1000,
            },
          },
          required: ["url"],
        },
      },
    ],
  };
});

// Handle tool calls
server.setRequestHandler(CallToolRequestSchema, async (request) => {
  if (request.params.name !== "screenshot") {
    throw new Error(`Unknown tool: ${request.params.name}`);
  }

  const args = request.params.arguments;
  const url = args.url;
  const fullPage = args.fullPage ?? false;
  const width = args.width ?? 1280;
  const height = args.height ?? 800;
  const waitForSelector = args.waitForSelector;
  const waitTime = args.waitTime ?? 1000;

  let browser;
  try {
    browser = await puppeteer.launch({
      headless: true,
      args: ["--no-sandbox", "--disable-setuid-sandbox"],
    });

    const page = await browser.newPage();
    await page.setViewport({ width, height });

    // Use 'load' for localhost (dev servers keep connections open), 'networkidle2' for others
    const isLocalhost = url.includes('localhost') || url.includes('127.0.0.1') || url.includes('10.0.0.');
    await page.goto(url, {
      waitUntil: isLocalhost ? "load" : "networkidle2",
      timeout: 30000,
    });

    // Wait for optional selector
    if (waitForSelector) {
      await page.waitForSelector(waitForSelector, { timeout: 10000 });
    }

    // Additional wait time for dynamic content
    if (waitTime > 0) {
      await new Promise((resolve) => setTimeout(resolve, waitTime));
    }

    // Get actual page dimensions
    const pageHeight = await page.evaluate(() => {
      return Math.max(
        document.body.scrollHeight,
        document.documentElement.scrollHeight
      );
    });

    // Check if we need to limit the height
    let wasTruncated = false;
    let actualCaptureHeight = height;
    let screenshotOptions = {
      encoding: "base64",
      type: "png",
    };

    if (fullPage) {
      if (pageHeight > MAX_SCREENSHOT_HEIGHT) {
        // Capture only up to max height using clip
        wasTruncated = true;
        actualCaptureHeight = MAX_SCREENSHOT_HEIGHT;
        screenshotOptions.clip = {
          x: 0,
          y: 0,
          width: width,
          height: MAX_SCREENSHOT_HEIGHT,
        };
      } else {
        screenshotOptions.fullPage = true;
        actualCaptureHeight = pageHeight;
      }
    }

    const screenshot = await page.screenshot(screenshotOptions);

    await browser.close();

    // Build response message
    let message = `Screenshot captured successfully from ${url} (${width}x${actualCaptureHeight}${fullPage ? ", full page" : ""})`;
    if (wasTruncated) {
      message += `\n\n⚠️ WARNING: The page height (${pageHeight}px) exceeded the maximum allowed height (${MAX_SCREENSHOT_HEIGHT}px). Only the top ${MAX_SCREENSHOT_HEIGHT}px of the page was captured. Consider creating a temporary web page for the section you are trying to view.`;
    }

    return {
      content: [
        {
          type: "image",
          data: screenshot,
          mimeType: "image/png",
        },
        {
          type: "text",
          text: message,
        },
      ],
    };
  } catch (error) {
    if (browser) {
      await browser.close();
    }
    return {
      content: [
        {
          type: "text",
          text: `Error capturing screenshot: ${error.message}`,
        },
      ],
      isError: true,
    };
  }
});

// Start the server
async function main() {
  const transport = new StdioServerTransport();
  await server.connect(transport);
  console.error("Screenshot MCP server running on stdio");
}

main().catch((error) => {
  console.error("Fatal error:", error);
  process.exit(1);
});
