import { NextRequest, NextResponse } from 'next/server';
import { getCachedSimulation, cacheSimulation } from '@/lib/simulationCache';

// Backend API URL - Flask backend runs on port 5329
const BACKEND_URL = process.env.BACKEND_URL || 'http://localhost:5329';

// Timeout for backend requests (in ms) - simulations can take a while
const BACKEND_TIMEOUT_MS = 300000; // 5 minutes

interface BackendResponse {
  success: boolean;
  error?: string;
  [key: string]: unknown;
}

// Transform backend response - only pass through real data, no placeholders
function transformBackendResponse(backendData: BackendResponse): Record<string, unknown> {
  // Pass through the backend response directly, just ensuring CCDF keys are strings
  const result: Record<string, unknown> = {
    success: true,
    num_simulations: backendData.num_simulations,
  };

  // Copy all data from backend, transforming CCDF keys where needed
  for (const [key, value] of Object.entries(backendData)) {
    if (key === 'success' || key === 'num_simulations') continue;

    if (value && typeof value === 'object') {
      result[key] = transformObjectCCDFKeys(value as Record<string, unknown>);
    } else {
      result[key] = value;
    }
  }

  return result;
}

// Recursively ensure CCDF keys are strings for frontend compatibility
function transformObjectCCDFKeys(obj: Record<string, unknown>): Record<string, unknown> {
  const result: Record<string, unknown> = {};

  for (const [key, value] of Object.entries(obj)) {
    if (value && typeof value === 'object' && !Array.isArray(value)) {
      // Check if this looks like a CCDF object (numeric keys)
      const keys = Object.keys(value as Record<string, unknown>);
      const hasNumericKeys = keys.some(k => !isNaN(Number(k)));

      if (hasNumericKeys && key.includes('ccdf')) {
        // Convert numeric keys to strings
        const transformed: Record<string, unknown> = {};
        for (const [k, v] of Object.entries(value as Record<string, unknown>)) {
          transformed[String(k)] = v;
        }
        result[key] = transformed;
      } else {
        // Recurse into nested objects
        result[key] = transformObjectCCDFKeys(value as Record<string, unknown>);
      }
    } else {
      result[key] = value;
    }
  }

  return result;
}

// POST handler - calls backend with caching
export async function POST(request: NextRequest): Promise<NextResponse> {
  try {
    const params = await request.json();
    console.log('[API] Black project simulation request received');

    // Check cache first
    const cached = getCachedSimulation<Record<string, unknown>>(params);
    if (cached) {
      console.log('[API] Returning cached result');
      return NextResponse.json(cached);
    }

    console.log('[API] Cache miss, calling backend...');

    // Build request for backend
    const backendRequest = {
      parameters: params,
      num_simulations: params.numSimulations || 100,
      time_range: [params.agreementYear || 2027, (params.agreementYear || 2027) + (params.numYearsToSimulate || 10)],
    };

    // Call backend with timeout
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), BACKEND_TIMEOUT_MS);

    try {
      const response = await fetch(`${BACKEND_URL}/api/run-black-project-simulation`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(backendRequest),
        signal: controller.signal,
      });

      clearTimeout(timeoutId);

      if (!response.ok) {
        const errorText = await response.text();
        console.error('[API] Backend error:', errorText);
        throw new Error(`Backend returned ${response.status}: ${errorText}`);
      }

      const backendData: BackendResponse = await response.json();

      if (!backendData.success) {
        throw new Error(backendData.error || 'Backend simulation failed');
      }

      // Log what we received
      console.log('[API] Backend response keys:', Object.keys(backendData));
      console.log('[API] num_simulations:', backendData.num_simulations);

      if (backendData.rate_of_computation) {
        const roc = backendData.rate_of_computation as Record<string, unknown>;
        console.log('[API] rate_of_computation keys:', Object.keys(roc));
        console.log('[API] initial_chip_stock_samples length:', (roc.initial_chip_stock_samples as unknown[])?.length);
      } else {
        console.log('[API] No rate_of_computation in response!');
      }

      // Transform backend response (only key transformations, no placeholders)
      const transformedData = transformBackendResponse(backendData);

      // Cache the result
      cacheSimulation(params, transformedData);

      console.log('[API] Backend call successful, returning data');
      return NextResponse.json(transformedData);

    } catch (fetchError: unknown) {
      clearTimeout(timeoutId);

      if (fetchError instanceof Error && fetchError.name === 'AbortError') {
        console.error('[API] Backend request timed out');
        return NextResponse.json(
          { success: false, error: 'Simulation timed out. Try reducing numSimulations.' },
          { status: 504 }
        );
      }

      // Check if backend is not running
      if (fetchError instanceof Error && fetchError.message.includes('ECONNREFUSED')) {
        console.error('[API] Backend not running');
        return NextResponse.json(
          { success: false, error: 'Backend server not running. Please start the Flask backend on port 5329.' },
          { status: 503 }
        );
      }

      throw fetchError;
    }

  } catch (error) {
    console.error('[API] Error:', error);
    const message = error instanceof Error ? error.message : 'Unknown error';
    return NextResponse.json(
      { success: false, error: message },
      { status: 500 }
    );
  }
}

// GET handler for health check
export async function GET(): Promise<NextResponse> {
  try {
    // Try to reach backend
    const response = await fetch(`${BACKEND_URL}/api/run-black-project-simulation`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ parameters: {}, num_simulations: 1, time_range: [2027, 2028] }),
      signal: AbortSignal.timeout(5000),
    });

    const backendStatus = response.ok ? 'connected' : 'error';

    return NextResponse.json({
      status: 'ok',
      backend: backendStatus,
      backendUrl: BACKEND_URL,
    });
  } catch {
    return NextResponse.json({
      status: 'ok',
      backend: 'disconnected',
      backendUrl: BACKEND_URL,
    });
  }
}
