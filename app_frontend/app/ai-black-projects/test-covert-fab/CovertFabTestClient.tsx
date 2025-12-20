'use client';

import dynamic from 'next/dynamic';
import { useState, useEffect } from 'react';
import { BlackProjectData } from '@/types/blackProject';
import '../ai-black-projects.css';

// Dynamically import CovertFabSection to avoid SSR issues with Math.random
const CovertFabSection = dynamic(
  () => import('../components/sections/CovertFabSection').then(mod => mod.CovertFabSection),
  { ssr: false }
);

export default function CovertFabTestClient() {
  const [data, setData] = useState<BlackProjectData | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Fetch data from dummy API endpoint
  useEffect(() => {
    async function fetchData() {
      try {
        setIsLoading(true);
        const response = await fetch('/api/black-project-dummy?agreement_year=2027');
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
          <strong>Data source:</strong> /api/black-project-dummy
        </div>
        {isLoading && <div className="text-blue-600 mt-2">Loading...</div>}
        {error && <div className="text-red-600 mt-2">Error: {error}</div>}
        {data && !isLoading && (
          <div className="text-green-600 mt-2">Data loaded successfully</div>
        )}
      </div>

      <CovertFabSection
        data={data}
        isLoading={isLoading}
        agreementYear={2027}
      />
    </div>
  );
}
