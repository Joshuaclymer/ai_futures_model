'use client';

import { useState, useCallback, useEffect, useRef, useLayoutEffect } from 'react';
import { createPortal } from 'react-dom';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';

// Tooltip content type - can be a string (markdown file name) or React node
export type TooltipContent = string | React.ReactNode;

// Trigger element rect for positioning
export interface TriggerRect {
  top: number;
  bottom: number;
  left: number;
  right: number;
  width: number;
  height: number;
}

// Tooltip state interface
interface TooltipState {
  visible: boolean;
  content: TooltipContent;
  triggerRect: TriggerRect | null;
}

// Cache for loaded markdown content
const markdownCache: Record<string, string> = {};

// Hook for managing tooltip state
export function useTooltip() {
  const [tooltipState, setTooltipState] = useState<TooltipState>({
    visible: false,
    content: null,
    triggerRect: null,
  });

  // Track whether mouse is over trigger or tooltip
  const isOverTrigger = useRef(false);
  const isOverTooltip = useRef(false);
  const hideTimeoutRef = useRef<NodeJS.Timeout | null>(null);

  const clearHideTimeout = useCallback(() => {
    if (hideTimeoutRef.current) {
      clearTimeout(hideTimeoutRef.current);
      hideTimeoutRef.current = null;
    }
  }, []);

  const scheduleHide = useCallback(() => {
    clearHideTimeout();
    // Small delay to allow moving between trigger and tooltip
    hideTimeoutRef.current = setTimeout(() => {
      if (!isOverTrigger.current && !isOverTooltip.current) {
        setTooltipState(prev => ({ ...prev, visible: false }));
      }
    }, 100);
  }, [clearHideTimeout]);

  const showTooltip = useCallback((content: TooltipContent, e: React.MouseEvent) => {
    clearHideTimeout();
    isOverTrigger.current = true;
    const rect = (e.currentTarget as HTMLElement).getBoundingClientRect();
    setTooltipState({
      visible: true,
      content,
      triggerRect: {
        top: rect.top,
        bottom: rect.bottom,
        left: rect.left,
        right: rect.right,
        width: rect.width,
        height: rect.height,
      },
    });
  }, [clearHideTimeout]);

  const hideTooltip = useCallback(() => {
    isOverTrigger.current = false;
    scheduleHide();
  }, [scheduleHide]);

  // Handlers for the tooltip element itself
  const onTooltipMouseEnter = useCallback(() => {
    clearHideTimeout();
    isOverTooltip.current = true;
  }, [clearHideTimeout]);

  const onTooltipMouseLeave = useCallback(() => {
    isOverTooltip.current = false;
    scheduleHide();
  }, [scheduleHide]);

  // Helper to create event handlers for a tooltip trigger
  const createTooltipHandlers = useCallback((content: TooltipContent) => ({
    onMouseEnter: (e: React.MouseEvent) => showTooltip(content, e),
    onMouseLeave: hideTooltip,
  }), [showTooltip, hideTooltip]);

  return {
    tooltipState,
    showTooltip,
    hideTooltip,
    createTooltipHandlers,
    onTooltipMouseEnter,
    onTooltipMouseLeave,
  };
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
  triggerRect: TriggerRect | null;
  onMouseEnter?: () => void;
  onMouseLeave?: () => void;
}

// Position calculation helper
type Position = 'right' | 'left' | 'above' | 'below';

interface CalculatedPosition {
  x: number;
  y: number;
  width: number;
  fits: boolean;
}

function calculatePosition(
  position: Position,
  triggerRect: TriggerRect,
  tooltipHeight: number,
  preferredWidth: number,
  padding: number,
  viewportWidth: number,
  viewportHeight: number
): CalculatedPosition {
  const gap = 10;
  let x = 0;
  let y = 0;
  let width = preferredWidth;

  switch (position) {
    case 'right':
      x = triggerRect.right + gap;
      width = Math.min(preferredWidth, viewportWidth - x - padding);
      // For tall content, start from top; otherwise try to center
      if (tooltipHeight > viewportHeight - padding * 2) {
        y = padding;
      } else {
        y = Math.max(padding, Math.min(
          triggerRect.top + triggerRect.height / 2 - tooltipHeight / 2,
          viewportHeight - tooltipHeight - padding
        ));
      }
      break;
    case 'left':
      width = Math.min(preferredWidth, triggerRect.left - gap - padding);
      x = triggerRect.left - gap - width;
      // For tall content, start from top; otherwise try to center
      if (tooltipHeight > viewportHeight - padding * 2) {
        y = padding;
      } else {
        y = Math.max(padding, Math.min(
          triggerRect.top + triggerRect.height / 2 - tooltipHeight / 2,
          viewportHeight - tooltipHeight - padding
        ));
      }
      break;
    case 'above':
      x = Math.max(padding, Math.min(
        triggerRect.left + triggerRect.width / 2 - width / 2,
        viewportWidth - width - padding
      ));
      y = triggerRect.top - gap - tooltipHeight;
      break;
    case 'below':
      x = Math.max(padding, Math.min(
        triggerRect.left + triggerRect.width / 2 - width / 2,
        viewportWidth - width - padding
      ));
      y = triggerRect.bottom + gap;
      break;
  }

  // Check if it fits
  const fits =
    x >= padding &&
    x + width <= viewportWidth - padding &&
    y >= padding &&
    y + tooltipHeight <= viewportHeight - padding &&
    width >= 200; // Minimum usable width

  return { x, y, width, fits };
}

export function Tooltip({ content, visible, triggerRect, onMouseEnter, onMouseLeave }: TooltipProps) {
  const tooltipRef = useRef<HTMLDivElement>(null);
  const contentRef = useRef<HTMLDivElement>(null);
  const [finalPosition, setFinalPosition] = useState<{ x: number; y: number; width: number; maxHeight: number | null }>({
    x: 0, y: 0, width: 400, maxHeight: null
  });
  const [measured, setMeasured] = useState(false);

  // Function to measure and position the tooltip
  const measureAndPosition = useCallback(() => {
    if (!contentRef.current || !triggerRect || typeof window === 'undefined') return;

    const viewportWidth = window.innerWidth;
    const viewportHeight = window.innerHeight;
    const padding = 20;
    const preferredWidth = 400;

    // Get the natural content height (content div has no constraints)
    const contentHeight = contentRef.current.scrollHeight + 30; // +30 for padding

    // Try positions in order of preference: right, left, below, above
    const positions: Position[] = ['right', 'left', 'below', 'above'];

    let bestPosition: CalculatedPosition | null = null;
    let bestAvailableHeight = 0;

    for (const pos of positions) {
      const calc = calculatePosition(
        pos,
        triggerRect,
        contentHeight,
        preferredWidth,
        padding,
        viewportWidth,
        viewportHeight
      );

      // Calculate available height for this position
      let availHeight = 0;
      if (pos === 'above') {
        availHeight = triggerRect.top - padding - 10;
      } else if (pos === 'below') {
        availHeight = viewportHeight - triggerRect.bottom - padding - 10;
      } else {
        // For left/right, use most of viewport height
        availHeight = viewportHeight - padding * 2;
      }

      if (calc.fits) {
        bestPosition = calc;
        bestAvailableHeight = availHeight;
        break;
      }

      // Keep track of the option with most available height
      if (!bestPosition || availHeight > bestAvailableHeight) {
        bestPosition = calc;
        bestAvailableHeight = availHeight;
      }
    }

    // If nothing fits perfectly, try with a wider tooltip to reduce height
    if (!bestPosition?.fits) {
      const wideWidth = Math.min(600, viewportWidth - padding * 2);
      for (const pos of positions) {
        const calc = calculatePosition(
          pos,
          triggerRect,
          contentHeight * 0.7, // Estimate reduced height with wider content
          wideWidth,
          padding,
          viewportWidth,
          viewportHeight
        );

        let availHeight = 0;
        if (pos === 'above') {
          availHeight = triggerRect.top - padding - 10;
        } else if (pos === 'below') {
          availHeight = viewportHeight - triggerRect.bottom - padding - 10;
        } else {
          availHeight = viewportHeight - padding * 2;
        }

        if (calc.fits) {
          bestPosition = calc;
          bestAvailableHeight = availHeight;
          break;
        }
      }
    }

    // Final fallback - position at top with full viewport height
    if (!bestPosition) {
      bestPosition = {
        x: padding,
        y: padding,
        width: Math.min(preferredWidth, viewportWidth - padding * 2),
        fits: false,
      };
      bestAvailableHeight = viewportHeight - padding * 2;
    }

    // Calculate max height: use available height, minimum 200px
    const maxHeight = Math.max(200, Math.min(bestAvailableHeight, viewportHeight - padding * 2));

    setFinalPosition({
      x: bestPosition.x,
      y: Math.max(padding, bestPosition.y),
      width: bestPosition.width,
      maxHeight: maxHeight,
    });
    setMeasured(true);
  }, [triggerRect]);

  // Initial positioning
  useLayoutEffect(() => {
    if (!visible || !triggerRect) {
      setMeasured(false);
      return;
    }

    // Initial measurement after first render
    const timer = requestAnimationFrame(measureAndPosition);
    return () => cancelAnimationFrame(timer);
  }, [visible, triggerRect, measureAndPosition]);

  // Re-measure when content size changes (e.g., async markdown loads)
  useEffect(() => {
    if (!visible || !contentRef.current) return;

    const observer = new ResizeObserver(() => {
      measureAndPosition();
    });

    observer.observe(contentRef.current);
    return () => observer.disconnect();
  }, [visible, measureAndPosition]);

  if (!visible || typeof window === 'undefined' || !content || !triggerRect) return null;

  // Determine if content is a markdown doc name (string) or React node
  const isMarkdownDoc = typeof content === 'string';

  return createPortal(
    <div
      ref={tooltipRef}
      className="bp-tooltip"
      onMouseEnter={onMouseEnter}
      onMouseLeave={onMouseLeave}
      style={{
        position: 'fixed',
        left: finalPosition.x,
        top: finalPosition.y,
        width: `${finalPosition.width}px`,
        maxWidth: 'calc(100vw - 40px)',
        maxHeight: finalPosition.maxHeight ? `${finalPosition.maxHeight}px` : 'calc(100vh - 40px)',
        overflowY: 'auto',
        backgroundColor: '#fffff8',
        color: '#333',
        padding: '15px',
        borderRadius: '6px',
        border: '1px solid #ddd',
        zIndex: 10000,
        fontSize: '13px',
        lineHeight: 1.6,
        boxShadow: '0 4px 16px rgba(0,0,0,0.12)',
        pointerEvents: 'auto',
        opacity: measured ? 1 : 0,
        transition: 'opacity 0.1s ease-in-out',
      }}
    >
      <div ref={contentRef}>
        {isMarkdownDoc ? (
          <MarkdownTooltipContent docName={content} />
        ) : (
          content
        )}
      </div>
    </div>,
    document.body
  );
}

// Export tooltip doc names as constants for type safety
export const TOOLTIP_DOCS = {
  // Compute parameters (compute_parameters.py)
  prc_capacity: 'compute_parameters/prc_capacity',
  transistor_density: 'compute_parameters/transistor_density',
  architecture_efficiency: 'compute_parameters/architecture_efficiency',
  dennard_scaling_end: 'compute_parameters/dennard_scaling_end',
  watts_per_tpp: 'compute_parameters/watts_per_tpp',
  watts_per_tpp_before_dennard: 'compute_parameters/watts_per_tpp_before_dennard',
  watts_per_tpp_after_dennard: 'compute_parameters/watts_per_tpp_after_dennard',
  energy_efficiency_improvement: 'compute_parameters/energy_efficiency_improvement',
  largest_ai_project: 'compute_parameters/largest_ai_project',
  prc_scanner_rampup: 'compute_parameters/prc_scanner_rampup',
  prc_sme_indigenization: 'compute_parameters/prc_sme_indigenization',
  chips_per_wafer: 'compute_parameters/chips_per_wafer',
  scanner_production_capacity: 'compute_parameters/scanner_production_capacity',
  fab_construction_time: 'compute_parameters/fab_construction_time',
  operating_labor_production: 'compute_parameters/operating_labor_production',
  ai_chip_lifespan: 'compute_parameters/ai_chip_lifespan',

  // Data center and energy parameters (data_center_and_energy_parameters.py)
  energy_efficiency: 'data_center_and_energy_parameters/energy_efficiency',
  prc_energy: 'data_center_and_energy_parameters/prc_energy',
  prc_energy_consumption: 'data_center_and_energy_parameters/prc_energy_consumption',
  mw_per_worker: 'data_center_and_energy_parameters/mw_per_worker',
  construction_workers: 'data_center_and_energy_parameters/construction_workers',
  h100_power: 'data_center_and_energy_parameters/h100_power',

  // Black project parameters (black_project_parameters.py)
  fraction_diverted: 'black_project_parameters/fraction_diverted',
  max_energy_proportion: 'black_project_parameters/max_energy_proportion',
  covert_unconcealed: 'black_project_parameters/covert_unconcealed',
  datacenter_start_year: 'black_project_parameters/datacenter_start_year',
  retrofitted_capacity: 'black_project_parameters/retrofitted_capacity',
  project_property: 'black_project_parameters/project_property',

  // Perceptions parameters (perceptions_parameters.py)
  prior_odds: 'perceptions_parameters/prior_odds',
  chip_stock_detection: 'perceptions_parameters/chip_stock_detection',
  energy_accounting_detection: 'perceptions_parameters/energy_accounting_detection',
  satellite_datacenter_detection: 'perceptions_parameters/satellite_datacenter_detection',
  sme_inventory_detection: 'perceptions_parameters/sme_inventory_detection',
  detection_time: 'perceptions_parameters/detection_time',
} as const;

export type TooltipDocName = keyof typeof TOOLTIP_DOCS;
