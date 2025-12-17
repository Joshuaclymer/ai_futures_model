import { useState, useRef, useEffect, type ReactNode } from 'react';
import { createPortal } from 'react-dom';

interface WithChartTooltipProps {
  explanation: string;
  children: ReactNode;
  className?: string;
  fullWidth?: boolean;
  tooltipPlacement?: 'left' | 'right';
}

export const WithChartTooltip = ({
  explanation,
  children,
  className,
  fullWidth = false,
  tooltipPlacement = 'right',
}: WithChartTooltipProps) => {
  const [showTooltip, setShowTooltip] = useState(false);
  const [tooltipPosition, setTooltipPosition] = useState({ top: 0, left: 0 });
  const triggerRef = useRef<HTMLSpanElement>(null);

  useEffect(() => {
    if (showTooltip && triggerRef.current) {
      const rect = triggerRef.current.getBoundingClientRect();
      const tooltipWidth = 256; // w-64 = 16rem = 256px
      const padding = 8;

      let left: number;
      if (tooltipPlacement === 'left') {
        left = rect.left - tooltipWidth - padding;
      } else {
        left = rect.right + padding;
      }

      // Prevent tooltip from going off-screen horizontally
      if (left + tooltipWidth > window.innerWidth - padding) {
        left = rect.left - tooltipWidth - padding;
      }
      if (left < padding) {
        left = padding;
      }

      // Position below the trigger, with some offset
      let top = rect.bottom + 4;

      // Prevent tooltip from going off-screen vertically
      if (top + 100 > window.innerHeight) {
        top = rect.top - 100 - 4;
      }

      setTooltipPosition({ top, left });
    }
  }, [showTooltip, tooltipPlacement]);

  const containerClasses = [
    'relative items-start gap-2 align-middle',
    fullWidth ? 'flex w-full justify-between' : 'inline-flex',
    className ?? '',
  ]
    .filter(Boolean)
    .join(' ');

  const labelClasses = fullWidth ? 'flex-1 leading-tight' : 'inline-flex leading-tight';

  return (
    <div className={containerClasses}>
      <span className={labelClasses}>{children}</span>
      <span
        ref={triggerRef}
        className="mt-[0.5px] relative inline-flex h-4 w-4 min-w-4 items-center justify-center rounded-full border border-gray-400 text-[10px] font-semibold leading-none text-gray-600 cursor-help transition-colors hover:border-gray-600 hover:text-gray-800 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-1 focus-visible:outline-gray-500"
        role="button"
        tabIndex={0}
        aria-label="Show explanation"
        aria-expanded={showTooltip}
        onMouseEnter={() => setShowTooltip(true)}
        onMouseLeave={() => setShowTooltip(false)}
        onFocus={() => setShowTooltip(true)}
        onBlur={() => setShowTooltip(false)}
      >
        i
      </span>
      {showTooltip && typeof document !== 'undefined' && createPortal(
        <div
          className="pointer-events-none fixed z-[9999]"
          style={{ top: tooltipPosition.top, left: tooltipPosition.left }}
        >
          <div className="w-64 rounded-md bg-gray-900 px-3 py-2 text-xs text-white shadow-lg font-system-mono font-normal">
            {explanation}
          </div>
        </div>,
        document.body
      )}
    </div>
  );
};
