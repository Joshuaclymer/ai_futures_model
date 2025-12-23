'use client';

import React, { Component, ErrorInfo, ReactNode } from 'react';

interface ErrorBoundaryProps {
  children: ReactNode;
  fallback?: ReactNode;
  /** Optional callback when an error occurs */
  onError?: (error: Error, errorInfo: ErrorInfo) => void;
}

interface ErrorBoundaryState {
  hasError: boolean;
  error: Error | null;
}

/**
 * ErrorBoundary - Catches JavaScript errors in child components and displays a fallback UI.
 *
 * Use this to wrap chart components or sections that might fail due to data issues
 * without crashing the entire page.
 *
 * Example:
 * ```tsx
 * <ErrorBoundary fallback={<div>Chart failed to load</div>}>
 *   <SomeChart data={data} />
 * </ErrorBoundary>
 * ```
 */
export class ErrorBoundary extends Component<ErrorBoundaryProps, ErrorBoundaryState> {
  constructor(props: ErrorBoundaryProps) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error: Error): ErrorBoundaryState {
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, errorInfo: ErrorInfo): void {
    console.error('ErrorBoundary caught an error:', error, errorInfo);
    this.props.onError?.(error, errorInfo);
  }

  render(): ReactNode {
    if (this.state.hasError) {
      if (this.props.fallback) {
        return this.props.fallback;
      }

      return (
        <div className="flex items-center justify-center h-full p-4 text-center">
          <div className="text-gray-500 text-sm">
            <p className="font-medium mb-1">Something went wrong</p>
            <p className="text-xs text-gray-400">
              {this.state.error?.message || 'An error occurred while rendering this component'}
            </p>
          </div>
        </div>
      );
    }

    return this.props.children;
  }
}

/**
 * ChartErrorBoundary - Specialized error boundary for chart components.
 * Shows a chart-specific error message.
 */
export function ChartErrorBoundary({ children }: { children: ReactNode }) {
  return (
    <ErrorBoundary
      fallback={
        <div className="flex items-center justify-center h-full bg-gray-50 rounded">
          <div className="text-gray-400 text-sm">
            Chart failed to render
          </div>
        </div>
      }
    >
      {children}
    </ErrorBoundary>
  );
}

export default ErrorBoundary;
