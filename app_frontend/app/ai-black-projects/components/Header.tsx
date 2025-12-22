'use client';

import Link from 'next/link';

// Header height constant - used for both the header and content margin
export const HEADER_HEIGHT = 56; // pixels

interface HeaderProps {
  onMenuClick: () => void;
}

export function Header({ onMenuClick }: HeaderProps) {
  return (
    <header
      className="fixed top-0 left-0 right-0 z-50 flex flex-row items-center justify-between px-4 lg:px-6"
      style={{ height: `${HEADER_HEIGHT}px`, backgroundColor: '#fffff8', borderBottom: '1px solid #e5e5e0' }}
    >
      {/* Mobile: hamburger menu on left */}
      <button
        onClick={onMenuClick}
        className="lg:hidden p-2 text-gray-600 hover:text-gray-900 hover:bg-gray-100 rounded"
        aria-label="Toggle parameters sidebar"
      >
        <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
        </svg>
      </button>

      <nav className="flex flex-row items-center h-full">
        <Link
          href="/ai-timelines-and-takeoff"
          className="px-2 lg:px-4 py-2 text-xs lg:text-sm text-gray-600 hover:text-gray-900 bp-nav-link"
        >
          AI Timelines and Takeoff
        </Link>
        <Link
          href="/ai-black-projects"
          className="px-2 lg:px-4 py-2 text-xs lg:text-sm text-gray-900 bp-nav-link"
        >
          Black Projects
        </Link>
      </nav>

      {/* Spacer for desktop to maintain layout */}
      <div className="hidden lg:block w-10" />
    </header>
  );
}
