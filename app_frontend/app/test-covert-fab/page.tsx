'use client';

import { CovertFabSection } from '../ai-black-projects/components/sections/CovertFabSection/index';
import '../ai-black-projects/ai-black-projects.css';

export default function TestCovertFabPage() {
  return (
    <div style={{ padding: '20px 40px', maxWidth: '1400px', margin: '0 auto' }}>
      <CovertFabSection data={null} agreementYear={2030} />
    </div>
  );
}
