'use client';

import { useState } from 'react';
import { useRouter } from 'next/navigation';
import { BlackProjectClient } from '@/app/ai-black-projects/BlackProjectClient';
import PlaygroundClient from '@/app/ai-timelines-and-takeoff/PlaygroundClient';

type TabType = 'timelines' | 'black-projects';

interface UnifiedAppClientProps {
  initialTab?: TabType;
  // Props for PlaygroundClient (only needed for timelines tab)
  benchmarkData?: any;
  initialComputeData?: any;
  initialParameters?: any;
  initialSampleTrajectories?: any[];
  initialSeed?: number;
}

export default function UnifiedAppClient({
  initialTab = 'timelines',
  benchmarkData,
  initialComputeData,
  initialParameters,
  initialSampleTrajectories,
  initialSeed,
}: UnifiedAppClientProps) {
  const [activeTab] = useState<TabType>(initialTab);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const router = useRouter();

  // Handle tab switching - navigate to the actual page for server-side data fetching
  const handleTabChange = (tab: TabType) => {
    if (tab === activeTab) return;
    const newPath = tab === 'timelines' ? '/ai-timelines-and-takeoff' : '/ai-black-projects';
    router.push(newPath);
  };

  return (
    <div className="min-h-screen bg-[#fffff8]">
      {/* Fixed Header with Tabs */}
      <header className="fixed top-0 left-0 right-0 z-50 flex flex-row items-center justify-between px-4 lg:px-6 py-0 border-b border-gray-200" style={{ backgroundColor: '#fffff8' }}>
        {/* Mobile hamburger button - only show for black-projects tab */}
        {activeTab === 'black-projects' && (
          <button
            type="button"
            onClick={() => setSidebarOpen(!sidebarOpen)}
            className="lg:hidden p-2 text-gray-600 hover:text-gray-900 hover:bg-gray-100 rounded"
            aria-label="Toggle parameters sidebar"
          >
            <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
            </svg>
          </button>
        )}
        {/* Spacer for when hamburger is not shown (timelines tab) */}
        {activeTab === 'timelines' && <div className="lg:hidden w-10" />}

        <nav className="flex flex-row">
          <button
            type="button"
            onClick={() => handleTabChange('timelines')}
            className={`px-2 lg:px-4 py-4 text-xs lg:text-sm font-medium transition-colors cursor-pointer ${
              activeTab === 'timelines'
                ? 'text-gray-900'
                : 'text-gray-600 hover:text-gray-900'
            }`}
            style={{ fontFamily: 'et-book, Georgia, serif' }}
          >
            AI Timelines and Takeoff
          </button>
          <button
            type="button"
            onClick={() => handleTabChange('black-projects')}
            className={`px-2 lg:px-4 py-4 text-xs lg:text-sm font-medium transition-colors cursor-pointer ${
              activeTab === 'black-projects'
                ? 'text-gray-900'
                : 'text-gray-600 hover:text-gray-900'
            }`}
            style={{ fontFamily: 'et-book, Georgia, serif' }}
          >
            Black Projects
          </button>
        </nav>

        {/* Spacer for desktop to maintain layout */}
        <div className="hidden lg:block w-10" />
      </header>

      {/* Mobile overlay when sidebar is open */}
      {sidebarOpen && activeTab === 'black-projects' && (
        <div
          className="fixed inset-0 bg-black bg-opacity-50 z-40 lg:hidden"
          style={{ top: '57px' }}
          onClick={() => setSidebarOpen(false)}
        />
      )}

      {/* Tab Content */}
      <div className="pt-[57px]">
        {/* Timelines Tab - conditionally render to avoid fixed positioning conflicts */}
        {activeTab === 'timelines' && (
          <PlaygroundClient
            benchmarkData={benchmarkData}
            initialComputeData={initialComputeData}
            initialParameters={initialParameters}
            initialSampleTrajectories={initialSampleTrajectories}
            initialSeed={initialSeed}
            hideHeader={true}
          />
        )}

        {/* Black Projects Tab - conditionally render to avoid fixed positioning conflicts */}
        {activeTab === 'black-projects' && (
          <BlackProjectClient
            initialData={null}
            hideHeader={true}
            externalSidebarOpen={sidebarOpen}
            onExternalSidebarClose={() => setSidebarOpen(false)}
          />
        )}
      </div>
    </div>
  );
}
