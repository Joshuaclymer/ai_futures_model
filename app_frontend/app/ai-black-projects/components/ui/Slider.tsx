'use client';

import { useState, useRef, useEffect } from 'react';
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

  return (
    <div className="bp-slider-group">
      {tooltipDoc ? (
        <label
          htmlFor={id}
          {...tooltipHandlers}
          style={{ cursor: 'help' }}
        >
          {label}<sup style={{ color: '#5E6FB8', marginLeft: '2px', fontSize: '9px' }}>?</sup>
        </label>
      ) : (
        <label htmlFor={id}>{label}</label>
      )}
      <input
        type="range"
        className="bp-slider"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
      />
      <div style={{ display: 'flex', justifyContent: 'center', marginTop: '2px' }}>
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
              width: '80px',
              padding: '3px 8px',
              fontSize: '10px',
              border: '1px solid #5E6FB8',
              borderRadius: '3px',
              textAlign: 'center',
              outline: 'none',
              boxShadow: '0 0 0 2px rgba(94, 111, 184, 0.2)',
            }}
          />
        ) : (
          <input
            id={id}
            type="text"
            value={displayValue}
            readOnly
            onFocus={handleFocus}
            onClick={handleStartEdit}
            style={{
              width: '80px',
              padding: '3px 8px',
              fontSize: '10px',
              border: 'none',
              textAlign: 'center',
              cursor: 'pointer',
              backgroundColor: 'transparent',
            }}
            title="Click to edit"
          />
        )}
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
