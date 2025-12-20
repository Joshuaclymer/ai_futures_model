'use client';

import { InitialStockSection } from '../components';
import '../ai-black-projects.css';

export default function InitialStockTestPage() {
  // Use only dummy data - no API call
  // Pass null for data so the component uses its internal dummy data generation
  return (
    <div className="min-h-screen bg-[#fffff8] p-6" style={{ maxWidth: 1200, margin: '0 auto' }}>
      <InitialStockSection
        data={null}
        isLoading={false}
        diversionProportion={0.1}
        agreementYear={2027}
      />
    </div>
  );
}
