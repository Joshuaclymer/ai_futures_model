import CovertFabTestClient from './CovertFabTestClient';

// Force dynamic rendering to avoid prerendering issues with Math.random
export const dynamic = 'force-dynamic';

export default function CovertFabTestPage() {
  return <CovertFabTestClient />;
}
