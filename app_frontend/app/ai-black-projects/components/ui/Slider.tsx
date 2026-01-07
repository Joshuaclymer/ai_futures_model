'use client';

import React, { useState, useRef, useEffect } from 'react';
import { useTooltip, Tooltip } from './Tooltip';

interface SliderProps {
  label: string;
  value: number;
  onChange: (v: number) => void;
  min: number;
  max: number;
  step: number;
  formatValue?: (v: number) => string;
  parseValue?: (s: string) => number;
  id?: string;
  tooltipDoc?: string; // Name of the markdown doc to show on hover (e.g., 'prior_odds')
}

export function Slider({
  label,
  value,
  onChange,
  min,
  max,
  step,
  formatValue,
  parseValue,
  id,
  tooltipDoc,
}: SliderProps) {
  const [isEditing, setIsEditing] = useState(false);
  const [editValue, setEditValue] = useState('');
  const inputRef = useRef<HTMLInputElement>(null);

  // Tooltip for parameter documentation
  const { tooltipState, createTooltipHandlers, onTooltipMouseEnter, onTooltipMouseLeave } = useTooltip();

  // Default formatting: round to 2 significant figures to avoid floating point display issues
  const defaultFormat = (v: number): string => {
    if (v === 0) return '0';
    const magnitude = Math.floor(Math.log10(Math.abs(v)));
    const precision = Math.max(0, 1 - magnitude); // 2 sig figs
    return v.toFixed(precision);
  };
  const displayValue = formatValue ? formatValue(value) : defaultFormat(value);

  // Auto-detect if this slider displays percentages based on formatValue output
  const isPercentageSlider = formatValue ? formatValue(0.5).includes('%') : false;

  // Default parser that handles common suffixes like %, x, years, GW, MW
  const defaultParseValue = (s: string): number => {
    // Remove common suffixes and whitespace
    const cleaned = s.trim()
      .replace(/\s*(years?|GW|MW|%|x)\s*$/i, '')
      .replace(/,/g, '');

    const numValue = parseFloat(cleaned);

    // If this is a percentage slider and user typed a number > 1 without %,
    // assume they meant it as a percentage (e.g., "50" means 50% = 0.5)
    if (isPercentageSlider && !s.includes('%') && numValue > 1) {
      return numValue / 100;
    }

    // Check if it was explicitly a percentage (user typed %)
    if (s.includes('%')) {
      return numValue / 100;
    }

    return numValue;
  };

  const handleStartEdit = () => {
    setIsEditing(true);
    setEditValue(displayValue);
  };

  const handleFinishEdit = () => {
    setIsEditing(false);
    const parser = parseValue || defaultParseValue;
    const newValue = parser(editValue);

    if (!isNaN(newValue)) {
      // Clamp to min/max and round to step
      const clamped = Math.max(min, Math.min(max, newValue));
      const rounded = Math.round(clamped / step) * step;
      // Only call onChange if the value actually changed
      if (rounded !== value) {
        onChange(rounded);
      }
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') {
      handleFinishEdit();
    } else if (e.key === 'Escape') {
      setIsEditing(false);
      setEditValue(displayValue);
    }
  };

  // Focus the input when entering edit mode
  useEffect(() => {
    if (isEditing && inputRef.current) {
      inputRef.current.focus();
      inputRef.current.select();
    }
  }, [isEditing]);

  // Handle external focus (from clicking parameter links)
  const handleFocus = () => {
    handleStartEdit();
  };

  // Get tooltip handlers if we have documentation
  const tooltipHandlers = tooltipDoc ? createTooltipHandlers(tooltipDoc) : null;

  // Calculate percentage for positioning value label below thumb
  const safeMin = typeof min === 'number' && !isNaN(min) ? min : 0;
  const safeMax = typeof max === 'number' && !isNaN(max) ? max : 100;
  const safeValue = typeof value === 'number' && !isNaN(value) ? value : safeMin;
  const percentage = ((safeValue - safeMin) / (safeMax - safeMin)) * 100;

  // Calculate label positioning to prevent overflow (matching timelines page)
  const getLabelStyle = (): React.CSSProperties => {
    if (percentage <= 15) {
      return { left: `${percentage}%`, transform: 'translateX(0)', whiteSpace: 'nowrap' };
    } else if (percentage >= 85) {
      return { left: `${percentage}%`, transform: 'translateX(-100%)', whiteSpace: 'nowrap' };
    } else {
      return { left: `${percentage}%`, transform: 'translateX(-50%)', whiteSpace: 'nowrap' };
    }
  };

  return (
    <div className="bp-slider-group">
      {tooltipDoc ? (
        <label
          htmlFor={id}
          {...tooltipHandlers}
          style={{ cursor: 'help' }}
        >
          {label}
          <sup style={{ color: '#5E6FB8', marginLeft: '2px', fontSize: '9px' }}>?</sup>
        </label>
      ) : (
        <label htmlFor={id}>{label}</label>
      )}
      <div style={{ display: 'flex', flexDirection: 'column' }}>
        <div style={{ position: 'relative' }}>
          <input
            type="range"
            id={id}
            className="bp-slider"
            min={min}
            max={max}
            step={step}
            value={value}
            onChange={(e) => onChange(parseFloat(e.target.value))}
          />
          {/* Value label positioned below thumb */}
          <div
            style={{
              position: 'absolute',
              top: '100%',
              marginTop: '2px',
              fontSize: '9px',
              color: 'rgba(0, 0, 0, 0.5)',
              fontWeight: 500,
              pointerEvents: isEditing ? 'auto' : 'none',
              ...getLabelStyle(),
            }}
          >
            {isEditing ? (
              <input
                ref={inputRef}
                id={id}
                type="text"
                value={editValue}
                onChange={(e) => setEditValue(e.target.value)}
                onBlur={handleFinishEdit}
                onKeyDown={handleKeyDown}
                style={{
                  width: '50px',
                  padding: '1px 3px',
                  fontSize: '9px',
                  border: '1px solid #333',
                  borderRadius: '2px',
                  outline: 'none',
                  textAlign: 'center',
                  pointerEvents: 'auto',
                }}
              />
            ) : (
              <span
                data-param-value={id}
                onClick={handleStartEdit}
                onFocus={handleFocus}
                tabIndex={0}
                style={{ cursor: 'text', pointerEvents: 'auto' }}
                title="Click to edit"
              >
                {displayValue}
              </span>
            )}
          </div>
        </div>
        {/* Spacer for the absolutely positioned value label */}
        <div style={{ height: '14px' }} />
      </div>
      {tooltipDoc && (
        <Tooltip
          content={tooltipState.content}
          visible={tooltipState.visible}
          triggerRect={tooltipState.triggerRect}
          onMouseEnter={onTooltipMouseEnter}
          onMouseLeave={onTooltipMouseLeave}
        />
      )}
    </div>
  );
}
