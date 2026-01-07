'use client';

import { CovertFabSection } from '../components/sections/CovertFabSection';
import { defaultParameters } from '../types';

// Generate years from 2030 to 2040
const years = Array.from({ length: 11 }, (_, i) => 2030 + i);

// Mock data for testing compute per month
const mockCovertFabData = {
  dashboard: {
    production: '15K H100e',
    energy: '0.5 GW',
    probFabBuilt: '56%',
    yearsOperational: '3.2',
    processNode: '28nm',
  },
  transistor_density: [
    { node: '28nm', density: 0.02, probability: 0.6, wattsPerTpp: 50 },
    { node: '14nm', density: 0.06, probability: 0.3, wattsPerTpp: 10 },
    { node: '7nm', density: 0.12, probability: 0.1, wattsPerTpp: 3 },
  ],
  individual_sim_points: [
    { density: 0.02, wattsPerTpp: 50 },
    { density: 0.06, wattsPerTpp: 10 },
    { density: 0.12, wattsPerTpp: 3 },
  ],
  chips_per_wafer: 28,
  architecture_efficiency: 0.85,
  h100_power: 700,
  is_operational: {
    years,
    median: [0, 0, 0.2, 0.5, 0.8, 0.9, 0.95, 1, 1, 1, 1],
    p25: [0, 0, 0.1, 0.3, 0.6, 0.7, 0.8, 0.9, 0.95, 1, 1],
    p75: [0, 0, 0.3, 0.7, 0.9, 1, 1, 1, 1, 1, 1],
  },
  wafer_starts_samples: [1000, 1200, 1500, 800, 1100, 1300, 900, 1400, 1000, 1250],
  // Monthly production data (should show constant rate once operational)
  compute_per_month: {
    years,
    median: [0, 0, 500, 1000, 1500, 1500, 1500, 1500, 1500, 1500, 1500],
    p25: [0, 0, 300, 700, 1000, 1000, 1000, 1000, 1000, 1000, 1000],
    p75: [0, 0, 800, 1500, 2000, 2000, 2000, 2000, 2000, 2000, 2000],
  },
  energy_per_month: {
    years,
    median: [0, 0, 0.1, 0.2, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3],
    p25: [0, 0, 0.05, 0.1, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2],
    p75: [0, 0, 0.15, 0.3, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4],
  },
};

export default function ComputeTestPage() {
  return (
    <div style={{ padding: '40px', backgroundColor: '#fffff8', maxWidth: '1200px', margin: '0 auto' }}>
      <h1 style={{ marginBottom: '20px' }}>CovertFabSection Test - Compute Per Month</h1>
      <p style={{ marginBottom: '20px', fontFamily: 'monospace' }}>
        Testing the &quot;Compute produced per month&quot; chart with mock monthly production data.
        <br />
        The chart should show a constant rate once operational (around 1500 H100e/month median),
        <br />
        NOT a cumulative increasing curve.
      </p>
      <CovertFabSection
        data={null}
        parameters={defaultParameters}
        covertFabData={mockCovertFabData}
      />
    </div>
  );
}
