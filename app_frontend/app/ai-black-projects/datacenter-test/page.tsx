import { Suspense } from 'react';
import DatacenterTestClient from './DatacenterTestClient';

export default function DatacenterTestPage() {
  return (
    <Suspense fallback={<div className="min-h-screen bg-[#fffff8] p-6">Loading...</div>}>
      <DatacenterTestClient />
    </Suspense>
  );
}
