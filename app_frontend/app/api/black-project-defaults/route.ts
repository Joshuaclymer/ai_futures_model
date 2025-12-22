import { NextResponse } from 'next/server';

// Backend API URL
const BACKEND_URL = process.env.BACKEND_URL || 'http://localhost:5329';

export async function GET(): Promise<NextResponse> {
  try {
    const response = await fetch(`${BACKEND_URL}/api/black-project-defaults`, {
      method: 'GET',
      signal: AbortSignal.timeout(10000),
    });

    if (!response.ok) {
      throw new Error(`Backend returned ${response.status}`);
    }

    const data = await response.json();

    if (!data.success) {
      throw new Error(data.error || 'Failed to get defaults');
    }

    return NextResponse.json(data);

  } catch (error) {
    console.error('[API] Error fetching defaults:', error);
    return NextResponse.json(
      { success: false, error: error instanceof Error ? error.message : 'Unknown error' },
      { status: 500 }
    );
  }
}
