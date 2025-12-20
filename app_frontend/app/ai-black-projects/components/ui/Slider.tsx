'use client';

interface SliderProps {
  label: string;
  value: number;
  onChange: (v: number) => void;
  min: number;
  max: number;
  step: number;
  formatValue?: (v: number) => string;
  id?: string;
}

export function Slider({
  label,
  value,
  onChange,
  min,
  max,
  step,
  formatValue,
  id,
}: SliderProps) {
  const displayValue = formatValue ? formatValue(value) : value.toString();
  return (
    <div className="bp-slider-group">
      <label htmlFor={id}>{label}</label>
      <input
        id={id}
        type="range"
        className="bp-slider"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
      />
      <div className="bp-slider-value">{displayValue}</div>
    </div>
  );
}
