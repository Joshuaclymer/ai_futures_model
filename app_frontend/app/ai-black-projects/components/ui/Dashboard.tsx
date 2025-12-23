'use client';

import React, { ReactNode } from 'react';
import './Dashboard.css';

interface DashboardProps {
  title?: string;
  children: ReactNode;
  width?: number | string;
  className?: string;
  style?: React.CSSProperties;
}

/**
 * Dashboard - Container for displaying key metrics in a sidebar layout.
 * Used across sections to show median outcomes and summary statistics.
 */
export function Dashboard({
  title = 'Median outcome',
  children,
  width = 240,
  className = '',
  style,
}: DashboardProps) {
  return (
    <div
      className={`bp-dashboard ${className}`}
      style={{ width, flexShrink: 0, padding: 20, ...style }}
    >
      {title && (
        <div
          className="plot-title"
          style={{
            textAlign: 'center',
            borderBottom: '1px solid #ddd',
            paddingBottom: 8,
            marginBottom: 20,
          }}
        >
          {title}
        </div>
      )}
      <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
        {children}
      </div>
    </div>
  );
}

interface DashboardItemProps {
  value: string | ReactNode;
  label: string;
  sublabel?: string;
  secondary?: string | ReactNode;
  size?: 'large' | 'small';
  valueColor?: string;
}

/**
 * DashboardItem - Individual metric display within a Dashboard.
 *
 * @param value - The primary value to display
 * @param label - Label describing the metric
 * @param sublabel - Optional smaller text below the label
 * @param secondary - Optional secondary value (e.g., energy equivalent)
 * @param size - 'large' for primary metrics, 'small' for secondary
 * @param valueColor - Custom color for the value
 */
export function DashboardItem({
  value,
  label,
  sublabel,
  secondary,
  size = 'large',
  valueColor,
}: DashboardItemProps) {
  const valueClass = size === 'large' ? 'bp-dashboard-value' : 'bp-dashboard-value-small';
  const labelClass = size === 'large' ? 'bp-dashboard-label' : 'bp-dashboard-label-light';

  return (
    <div className="bp-dashboard-item">
      <div className={valueClass} style={valueColor ? { color: valueColor } : undefined}>
        {value}
      </div>
      {secondary && (
        <div style={{ fontSize: 18, color: '#666', marginBottom: 5 }}>{secondary}</div>
      )}
      <div className={labelClass}>{label}</div>
      {sublabel && <div className="bp-dashboard-sublabel">{sublabel}</div>}
    </div>
  );
}

export default Dashboard;
