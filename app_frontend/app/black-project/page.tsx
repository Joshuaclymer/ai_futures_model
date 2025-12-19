import { BlackProjectClient } from './BlackProjectClient';

export default function BlackProjectPage() {
  // Data fetching is done client-side to avoid SSR issues with local backend
  return <BlackProjectClient initialData={null} />;
}

export const metadata = {
  title: 'Covert Compute Production Model',
  description: 'Monte Carlo simulation of covert compute production capabilities',
};
