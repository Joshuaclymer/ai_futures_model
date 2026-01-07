import { NextRequest, NextResponse } from 'next/server';
import fs from 'fs';
import path from 'path';

export async function GET(
  request: NextRequest,
  { params }: { params: Promise<{ path: string[] }> }
) {
  const { path: pathSegments } = await params;
  const filePath = pathSegments.join('/');

  // Construct the full path to the file
  const docsPath = path.join(
    process.cwd(),
    '..',
    'ai_futures_simulator',
    'parameters',
    'parameter_documentation',
    filePath
  );

  try {
    // Check if file exists
    if (!fs.existsSync(docsPath)) {
      return NextResponse.json({ error: 'File not found' }, { status: 404 });
    }

    const content = fs.readFileSync(docsPath);
    const ext = path.extname(filePath).toLowerCase();

    // Set content type based on file extension
    const contentTypes: Record<string, string> = {
      '.html': 'text/html',
      '.png': 'image/png',
      '.jpg': 'image/jpeg',
      '.jpeg': 'image/jpeg',
      '.gif': 'image/gif',
      '.svg': 'image/svg+xml',
      '.css': 'text/css',
      '.js': 'application/javascript',
      '.csv': 'text/csv',
    };

    const contentType = contentTypes[ext] || 'application/octet-stream';

    return new NextResponse(content, {
      headers: {
        'Content-Type': contentType,
        'Cache-Control': 'no-cache, no-store, must-revalidate',
      },
    });
  } catch (error) {
    console.error('Error serving parameter doc:', error);
    return NextResponse.json({ error: 'Failed to read file' }, { status: 500 });
  }
}
