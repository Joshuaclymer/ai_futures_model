'use client';

import { useState, useEffect } from 'react';
import dynamic from 'next/dynamic';
import { BlackProjectClient } from '@/app/ai-black-projects/BlackProjectClient';

// Dynamically import PlaygroundClient (heavy component for timelines)
const PlaygroundClient = dynamic(() => import('@/app/ai-timelines-and-takeoff/PlaygroundClient'), {
  ssr: false,
  loading: () => <TabLoadingPlaceholder tab="timelines" />,
});

type TabType = 'timelines' | 'black-projects';

interface UnifiedAppClientProps {
  initialTab?: TabType;
  // Props for PlaygroundClient
  benchmarkData: any;
  initialComputeData: any;
  initialParameters: any;
  initialSampleTrajectories: any[];
  initialSeed: number;
}

// Loading placeholder component
function TabLoadingPlaceholder({ tab }: { tab: TabType }) {
  return (
    <div className="flex items-center justify-center h-[calc(100vh-57px)] bg-[#fffff8]">
      <div className="flex flex-col items-center gap-4">
        <div className="w-12 h-12 border-4 border-gray-200 border-t-blue-500 rounded-full animate-spin" />
        <span className="text-sm text-gray-500">
          Loading {tab === 'timelines' ? 'AI Timelines' : 'Black Projects'}...
        </span>
      </div>
    </div>
  );
}

export default function UnifiedAppClient({
  initialTab = 'timelines',
  benchmarkData,
  initialComputeData,
  initialParameters,
  initialSampleTrajectories,
  initialSeed,
}: UnifiedAppClientProps) {
  const [activeTab, setActiveTab] = useState<TabType>(initialTab);
  const [blackProjectsLoaded, setBlackProjectsLoaded] = useState(initialTab === 'black-projects');
  const [timelinesLoaded, setTimelinesLoaded] = useState(initialTab === 'timelines');

  // Handle tab switching - set loaded state synchronously to avoid empty flash
  const handleTabChange = (tab: TabType) => {
    if (tab === 'black-projects' && !blackProjectsLoaded) {
      setBlackProjectsLoaded(true);
    }
    if (tab === 'timelines' && !timelinesLoaded) {
      setTimelinesLoaded(true);
    }
    setActiveTab(tab);
  };

  // Update URL without navigation (for bookmarking/sharing)
  useEffect(() => {
    const newPath = activeTab === 'timelines' ? '/ai-timelines-and-takeoff' : '/ai-black-projects';
    window.history.replaceState(null, '', newPath);
  }, [activeTab]);

  return (
    <div className="min-h-screen bg-[#fffff8]">
      {/* Fixed Header with Tabs */}
      <header className="fixed top-0 left-0 right-0 z-50 flex flex-row items-center justify-between px-6 py-0 border-b border-gray-200 bg-white">
        <nav className="flex flex-row">
          <button
            type="button"
            onClick={() => handleTabChange('timelines')}
            className={`px-4 py-4 text-sm font-medium transition-colors cursor-pointer ${
              activeTab === 'timelines'
                ? 'text-gray-900'
                : 'text-gray-600 hover:text-gray-900 hover:bg-gray-50'
            }`}
            style={{ fontFamily: 'et-book, Georgia, serif' }}
          >
            AI Timelines and Takeoff
          </button>
          <button
            type="button"
            onClick={() => handleTabChange('black-projects')}
            className={`px-4 py-4 text-sm font-medium transition-colors cursor-pointer ${
              activeTab === 'black-projects'
                ? 'text-gray-900'
                : 'text-gray-600 hover:text-gray-900 hover:bg-gray-50'
            }`}
            style={{ fontFamily: 'et-book, Georgia, serif' }}
          >
            Black Projects
          </button>
        </nav>
      </header>

      {/* Tab Content */}
      <div className="pt-[57px]">
        {/* Timelines Tab - keep mounted but hidden for instant switching */}
        <div style={{ display: activeTab === 'timelines' ? 'block' : 'none' }}>
          {timelinesLoaded && (
            <PlaygroundClient
              benchmarkData={benchmarkData}
              initialComputeData={initialComputeData}
              initialParameters={initialParameters}
              initialSampleTrajectories={initialSampleTrajectories}
              initialSeed={initialSeed}
              hideHeader={true}
            />
          )}
        </div>

        {/* Black Projects Tab - keep mounted but hidden for instant switching */}
        <div style={{ display: activeTab === 'black-projects' ? 'block' : 'none' }}>
          {blackProjectsLoaded ? (
            <BlackProjectClient initialData={null} hideHeader={true} />
          ) : (
            <TabLoadingPlaceholder tab="black-projects" />
          )}
        </div>
      </div>
    </div>
  );
}
