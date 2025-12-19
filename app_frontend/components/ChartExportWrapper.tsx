'use client';

import { ReactNode, forwardRef } from 'react';
import { EXPORT_COLORS } from '@/utils/chartExport';

export interface ChartExportWrapperProps {
  title: string;
  subtitle?: string;
  children: ReactNode;
  width: number;
  height: number;
  showBranding?: boolean;
  brandingUrl?: string;
}

/**
 * Wrapper component that adds standalone context for chart exports
 * Includes title, subtitle, and optional branding footer
 */
export const ChartExportWrapper = forwardRef<HTMLDivElement, ChartExportWrapperProps>(({
  title,
  subtitle,
  children,
  width,
  height,
  showBranding = true,
  brandingUrl = 'aifuturesmodel.com',
}, ref) => {
  const titleHeight = 40;
  const subtitleHeight = subtitle ? 24 : 0;
  const brandingHeight = showBranding ? 28 : 0;
  const contentHeight = height - titleHeight - subtitleHeight - brandingHeight;

  return (
    <div
      ref={ref}
      style={{
        width,
        height,
        backgroundColor: EXPORT_COLORS.background,
        color: EXPORT_COLORS.foreground,
        fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, monospace',
        display: 'flex',
        flexDirection: 'column',
        padding: '12px 16px',
        boxSizing: 'border-box',
      }}
    >
      {/* Title */}
      <div
        style={{
          height: titleHeight,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
        }}
      >
        <h1
          style={{
            fontSize: '18px',
            fontWeight: 600,
            margin: 0,
            color: EXPORT_COLORS.foreground,
          }}
        >
          {title}
        </h1>
      </div>

      {/* Subtitle */}
      {subtitle && (
        <div
          style={{
            height: subtitleHeight,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
          }}
        >
          <p
            style={{
              fontSize: '12px',
              margin: 0,
              color: '#666666',
            }}
          >
            {subtitle}
          </p>
        </div>
      )}

      {/* Content */}
      <div
        style={{
          flex: 1,
          minHeight: contentHeight,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
        }}
      >
        {children}
      </div>

      {/* Branding footer */}
      {showBranding && (
        <div
          style={{
            height: brandingHeight,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
          }}
        >
          <span
            style={{
              fontSize: '11px',
              color: '#888888',
            }}
          >
            {brandingUrl}
          </span>
        </div>
      )}
    </div>
  );
});

ChartExportWrapper.displayName = 'ChartExportWrapper';
