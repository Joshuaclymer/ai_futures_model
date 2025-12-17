'use client';

import { ReactNode } from 'react';

interface ChartLoadingOverlayProps {
  isLoading: boolean;
  children: ReactNode;
  className?: string;
}

export function ChartLoadingOverlay({ isLoading, children, className }: ChartLoadingOverlayProps) {
  const wrapperClassName = ['relative', className].filter(Boolean).join(' ');

  return (
    <div className={wrapperClassName}>
      {children}
      {isLoading && (
        <div className="absolute inset-0 z-20 flex items-center justify-center bg-vivid-background/70 backdrop-blur-[1px]">
          <span
            className="h-6 w-6 animate-spin rounded-full border-2 border-gray-400 border-t-transparent"
            aria-hidden="true"
          />
        </div>
      )}
    </div>
  );
}


