import { NextRequest, NextResponse } from 'next/server';
import fs from 'fs';
import path from 'path';

// API route to serve parameter documentation markdown files
export async function GET(
  request: NextRequest,
  { params }: { params: Promise<{ name: string }> }
) {
  const { name } = await params;

  // Sanitize the name to prevent directory traversal
  const sanitizedName = name.replace(/[^a-zA-Z0-9_-]/g, '');

  // Path to the parameter documentation directory
  const docPath = path.join(
    process.cwd(),
    '..',
    'ai_futures_simulator',
    'parameters',
    'parameter_documentation',
    `${sanitizedName}.md`
  );

  try {
    const content = fs.readFileSync(docPath, 'utf-8');
    return NextResponse.json({ content });
  } catch (error) {
    // Return a default message if file not found
    return NextResponse.json(
      { content: `# ${sanitizedName}\n\nDocumentation not available.` },
      { status: 404 }
    );
  }
}
