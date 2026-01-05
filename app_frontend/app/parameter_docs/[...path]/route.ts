import { NextRequest, NextResponse } from 'next/server';
import { readFile } from 'fs/promises';
import path from 'path';

// Serve files from ai_futures_simulator/parameters/parameter_documentation
const DOCS_DIR = path.join(process.cwd(), '..', 'ai_futures_simulator', 'parameters', 'parameter_documentation');

// MIME types for supported file types
const MIME_TYPES: Record<string, string> = {
  '.md': 'text/markdown; charset=utf-8',
  '.png': 'image/png',
  '.jpg': 'image/jpeg',
  '.jpeg': 'image/jpeg',
  '.gif': 'image/gif',
  '.svg': 'image/svg+xml',
  '.webp': 'image/webp',
};

export async function GET(
  request: NextRequest,
  { params }: { params: Promise<{ path: string[] }> }
) {
  const { path: pathSegments } = await params;
  const fileName = pathSegments.join('/');

  // Sanitize: only allow alphanumeric, underscores, hyphens, dots, and forward slashes
  // Prevent directory traversal
  if (fileName.includes('..') || !/^[a-zA-Z0-9_\-./]+$/.test(fileName)) {
    return NextResponse.json({ error: 'Invalid path' }, { status: 400 });
  }

  const filePath = path.join(DOCS_DIR, fileName);

  // Ensure the resolved path is within DOCS_DIR
  const resolvedPath = path.resolve(filePath);
  const resolvedDocsDir = path.resolve(DOCS_DIR);
  if (!resolvedPath.startsWith(resolvedDocsDir)) {
    return NextResponse.json({ error: 'Invalid path' }, { status: 400 });
  }

  const ext = path.extname(fileName).toLowerCase();
  const mimeType = MIME_TYPES[ext];

  if (!mimeType) {
    return NextResponse.json({ error: 'Unsupported file type' }, { status: 400 });
  }

  try {
    const content = await readFile(filePath);
    return new NextResponse(content, {
      headers: {
        'Content-Type': mimeType,
        'Cache-Control': 'public, max-age=3600', // Cache for 1 hour
      },
    });
  } catch (error) {
    return NextResponse.json({ error: 'File not found' }, { status: 404 });
  }
}
