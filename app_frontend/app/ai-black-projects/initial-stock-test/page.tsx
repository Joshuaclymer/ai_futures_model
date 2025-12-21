'use client';

import dynamic from 'next/dynamic';
import '../ai-black-projects.css';
import { defaultParameters } from '../types';

// Dynamically import to avoid SSR issues with Math.random()
const InitialStockSection = dynamic(
  () => import('../components/sections/InitialStockSection').then(mod => mod.InitialStockSection),
  { ssr: false }
);

export default function InitialStockTestPage() {
  // Use only dummy data - no API call
  // Pass null for data so the component uses its internal dummy data generation
  return (
    <div className="min-h-screen bg-[#fffff8] p-6" style={{ maxWidth: 1200, margin: '0 auto' }}>
      <InitialStockSection
        data={null}
        isLoading={false}
        parameters={defaultParameters}
      />
    </div>
  );
}
