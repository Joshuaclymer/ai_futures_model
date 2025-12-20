'use client';

import { useState, useCallback, useEffect } from 'react';
import { createPortal } from 'react-dom';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';

// Tooltip content type - can be a string (markdown file name) or React node
export type TooltipContent = string | React.ReactNode;

// Tooltip state interface
interface TooltipState {
  visible: boolean;
  content: TooltipContent;
  position: { x: number; y: number };
}

// Cache for loaded markdown content
const markdownCache: Record<string, string> = {};

// Hook for managing tooltip state
export function useTooltip() {
  const [tooltipState, setTooltipState] = useState<TooltipState>({
    visible: false,
    content: null,
    position: { x: 0, y: 0 },
  });

  const showTooltip = useCallback((content: TooltipContent, e: React.MouseEvent) => {
    const rect = (e.currentTarget as HTMLElement).getBoundingClientRect();
    const viewportWidth = window.innerWidth;

    // Position tooltip to the right of element, or left if not enough space
    let x = rect.right + 10;
    if (x + 420 > viewportWidth) {
      x = Math.max(10, rect.left - 420);
    }

    // Keep tooltip within vertical viewport bounds
    const y = Math.max(20, Math.min(rect.top, window.innerHeight - 300));

    setTooltipState({
      visible: true,
      content,
      position: { x, y },
    });
  }, []);

  const hideTooltip = useCallback(() => {
    setTooltipState(prev => ({ ...prev, visible: false }));
  }, []);

  // Helper to create event handlers for a tooltip
  const createTooltipHandlers = useCallback((content: TooltipContent) => ({
    onMouseEnter: (e: React.MouseEvent) => showTooltip(content, e),
    onMouseLeave: hideTooltip,
  }), [showTooltip, hideTooltip]);

  return { tooltipState, showTooltip, hideTooltip, createTooltipHandlers };
}

// Markdown tooltip content component
function MarkdownTooltipContent({ docName }: { docName: string }) {
  const [markdown, setMarkdown] = useState<string | null>(markdownCache[docName] || null);
  const [loading, setLoading] = useState(!markdownCache[docName]);

  useEffect(() => {
    if (markdownCache[docName]) {
      setMarkdown(markdownCache[docName]);
      setLoading(false);
      return;
    }

    setLoading(true);
    fetch(`/parameter_docs/${docName}.md`)
      .then(res => {
        if (!res.ok) throw new Error('Not found');
        return res.text();
      })
      .then(content => {
        markdownCache[docName] = content;
        setMarkdown(content);
      })
      .catch(() => {
        setMarkdown(`# ${docName}\n\nFailed to load documentation.`);
      })
      .finally(() => {
        setLoading(false);
      });
  }, [docName]);

  if (loading) {
    return <div style={{ color: '#666', fontStyle: 'italic' }}>Loading...</div>;
  }

  return (
    <ReactMarkdown
      remarkPlugins={[remarkGfm]}
      components={{
        // Custom styling for markdown elements
        h1: ({ children }) => (
          <h1 style={{ fontSize: '16px', fontWeight: 'bold', marginBottom: '10px', marginTop: 0 }}>
            {children}
          </h1>
        ),
        h2: ({ children }) => (
          <h2 style={{ fontSize: '14px', fontWeight: 'bold', marginBottom: '8px', marginTop: '12px' }}>
            {children}
          </h2>
        ),
        p: ({ children }) => (
          <p style={{ marginBottom: '8px', marginTop: 0 }}>{children}</p>
        ),
        ul: ({ children }) => (
          <ul style={{ marginLeft: '16px', marginBottom: '8px' }}>{children}</ul>
        ),
        li: ({ children }) => (
          <li style={{ marginBottom: '4px' }}>{children}</li>
        ),
        code: ({ children, className }) => {
          const isBlock = className?.includes('language-');
          if (isBlock) {
            return (
              <pre style={{
                background: '#f5f5f5',
                padding: '8px',
                borderRadius: '4px',
                overflow: 'auto',
                fontSize: '12px',
                marginBottom: '8px',
              }}>
                <code>{children}</code>
              </pre>
            );
          }
          return (
            <code style={{
              background: '#f5f5f5',
              padding: '2px 4px',
              borderRadius: '2px',
              fontSize: '12px',
            }}>
              {children}
            </code>
          );
        },
        strong: ({ children }) => (
          <strong style={{ fontWeight: 'bold' }}>{children}</strong>
        ),
      }}
    >
      {markdown || ''}
    </ReactMarkdown>
  );
}

// Tooltip render component
interface TooltipProps {
  content: TooltipContent;
  visible: boolean;
  position: { x: number; y: number };
}

export function Tooltip({ content, visible, position }: TooltipProps) {
  if (!visible || typeof window === 'undefined' || !content) return null;

  // Determine if content is a markdown doc name (string) or React node
  const isMarkdownDoc = typeof content === 'string';

  return createPortal(
    <div
      className="bp-tooltip"
      style={{
        position: 'fixed',
        left: position.x,
        top: position.y,
        maxWidth: '400px',
        maxHeight: '80vh',
        overflowY: 'auto',
        backgroundColor: '#fff',
        color: '#333',
        padding: '15px',
        borderRadius: '6px',
        border: '1px solid #ddd',
        zIndex: 10000,
        fontSize: '13px',
        lineHeight: 1.6,
        boxShadow: '0 4px 16px rgba(0,0,0,0.12)',
        pointerEvents: 'none',
      }}
    >
      {isMarkdownDoc ? (
        <MarkdownTooltipContent docName={content} />
      ) : (
        content
      )}
    </div>,
    document.body
  );
}

// Export tooltip doc names as constants for type safety
export const TOOLTIP_DOCS = {
  // Datacenter tooltips
  prc_capacity: 'prc_capacity',
  fraction_diverted: 'fraction_diverted',
  retrofitted_capacity: 'retrofitted_capacity',
  prc_energy: 'prc_energy',
  prc_energy_consumption: 'prc_energy_consumption',
  max_energy_proportion: 'max_energy_proportion',
  covert_unconcealed: 'covert_unconcealed',
  construction_workers: 'construction_workers',
  mw_per_worker: 'mw_per_worker',
  datacenter_start_year: 'datacenter_start_year',
  // Covert fab tooltips
  fab_construction_time: 'fab_construction_time',
  operating_labor_production: 'operating_labor_production',
  chips_per_wafer: 'chips_per_wafer',
  transistor_density: 'transistor_density',
  architecture_efficiency: 'architecture_efficiency',
  watts_per_tpp: 'watts_per_tpp',
  watts_per_tpp_before_dennard: 'watts_per_tpp_before_dennard',
  watts_per_tpp_after_dennard: 'watts_per_tpp_after_dennard',
  h100_power: 'h100_power',
  ai_chip_lifespan: 'ai_chip_lifespan',
  largest_ai_project: 'largest_ai_project',
  prc_scanner_rampup: 'prc_scanner_rampup',
  prc_sme_indigenization: 'prc_sme_indigenization',
  dennard_scaling_end: 'dennard_scaling_end',
  scanner_production_capacity: 'scanner_production_capacity',
  // Detection tooltips
  prior_odds: 'prior_odds',
  energy_efficiency: 'energy_efficiency',
  energy_efficiency_improvement: 'energy_efficiency_improvement',
  satellite_datacenter_detection: 'satellite_datacenter_detection',
  energy_accounting_detection: 'energy_accounting_detection',
  chip_stock_detection: 'chip_stock_detection',
  sme_inventory_detection: 'sme_inventory_detection',
  detection_time: 'detection_time',
  project_property: 'project_property',
} as const;

export type TooltipDocName = keyof typeof TOOLTIP_DOCS;
