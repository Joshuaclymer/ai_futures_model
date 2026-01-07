'use client';

import { CovertFabSection } from '../components/sections/CovertFabSection';
import { defaultParameters } from '../types';

// Mock data for testing - covert fab section data
const mockCovertFabData = {
  dashboard: {
    production: '15K H100e',
    energy: '0.5 GW',
    probFabBuilt: '56%',
    yearsOperational: '3.2',
    processNode: '28nm',
  },
  transistor_density: [
    { node: '28nm', density: 0.02, probability: 0.6 },
    { node: '14nm', density: 0.06, probability: 0.3 },
    { node: '7nm', density: 0.12, probability: 0.1 },
  ],
  individual_sim_points: [
    { density: 0.02, wattsPerTpp: 50 },
    { density: 0.06, wattsPerTpp: 10 },
    { density: 0.12, wattsPerTpp: 3 },
  ],
  chips_per_wafer: 28,
  architecture_efficiency: 0.85,
  h100_power: 700,
};

// Use default parameters but override the Dennard scaling value
const mockParameters = {
  ...defaultParameters,
  transistorDensityAtEndOfDennardScaling: 2.94,
};

export default function DennardTestPage() {
  return (
    <div style={{ padding: '40px', backgroundColor: '#fffff8', maxWidth: '1200px', margin: '0 auto' }}>
      <h1 style={{ marginBottom: '20px' }}>CovertFabSection Test - Dennard Scaling Line Position</h1>
      <p style={{ marginBottom: '20px', fontFamily: 'monospace' }}>
        Check that the dotted line in the &quot;Watts per TPP&quot; chart is at the kink of the curve.
      </p>
      <CovertFabSection
        data={null}
        parameters={mockParameters}
        covertFabData={mockCovertFabData}
      />
    </div>
  );
}
