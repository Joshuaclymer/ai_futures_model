'use client';

import dynamic from 'next/dynamic';
import { useState, useEffect } from 'react';
import { SimulationData } from '../types';
import '../ai-black-projects.css';
import { defaultParameters } from '../types';

// Dynamically import to avoid SSR issues
const DatacenterSection = dynamic(
  () => import('../components/sections/DatacenterSection').then(mod => mod.DatacenterSection),
  { ssr: false }
);

export default function DatacenterTestClient() {
  const [data, setData] = useState<SimulationData | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [parameters] = useState(defaultParameters);

  // Fetch data from Flask backend dummy API endpoint
  useEffect(() => {
    async function fetchData() {
      try {
        setIsLoading(true);
        const BACKEND_URL = process.env.NEXT_PUBLIC_BACKEND_URL || 'http://localhost:5329';
        const response = await fetch(`${BACKEND_URL}/api/black-project-dummy?ai_slowdown_start_year=2027`);
        if (!response.ok) {
          throw new Error(`API error: ${response.status}`);
        }
        const result = await response.json();
        console.log('API Response:', result);
        setData(result);
      } catch (err) {
        console.error('Error fetching data:', err);
        setError(err instanceof Error ? err.message : 'Unknown error');
      } finally {
        setIsLoading(false);
      }
    }
    fetchData();
  }, []);

  return (
    <div className="min-h-screen bg-[#fffff8] p-6">
      {/* Status info */}
      <div className="mb-6 p-4 bg-white border rounded">
        <div className="text-sm text-gray-600">
          <strong>Data source:</strong> Flask backend at localhost:5329/api/black-project-dummy
        </div>
        {isLoading && <div className="text-blue-600 mt-2">Loading...</div>}
        {error && <div className="text-red-600 mt-2">Error: {error}</div>}
        {data && !isLoading && (
          <div className="text-green-600 mt-2">Data loaded successfully</div>
        )}
      </div>

      <DatacenterSection
        data={data}
        isLoading={isLoading}
        parameters={parameters}
      />
    </div>
  );
}
