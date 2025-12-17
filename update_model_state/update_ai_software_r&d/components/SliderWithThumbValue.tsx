import React from 'react';

interface SliderWithThumbValueProps {
  value: number;
  min: number;
  max: number;
  step: string | number;
  onChange: (value: number) => void;
  onMouseDown?: () => void;
  onTouchStart?: () => void;
  onMouseUp?: () => void;
  onTouchEnd?: () => void;
  onKeyDown?: (event: React.KeyboardEvent<HTMLInputElement>) => void;
  onKeyUp?: (event: React.KeyboardEvent<HTMLInputElement>) => void;
  onMouseEnter?: () => void;
  onMouseLeave?: () => void;
  onFocus?: () => void;
  onBlur?: () => void;
  displayValue: string;
  className?: string;
  formatValue?: (value: number) => string;
  useLogScale?: boolean;
  stepCount?: number;
}

export const SliderWithThumbValue: React.FC<SliderWithThumbValueProps> = ({
  value,
  min,
  max,
  step,
  onChange,
  onMouseDown,
  onTouchStart,
  onMouseUp,
  onTouchEnd,
  onKeyDown,
  onKeyUp,
  onMouseEnter,
  onMouseLeave,
  onFocus,
  onBlur,
  displayValue,
  className = "w-full rounded-lg appearance-none cursor-pointer slider",
  formatValue,
  useLogScale = false,
  stepCount,
}) => {
  const canUseLogScale = useLogScale && min > 0 && max > min;
  
  if (useLogScale && !canUseLogScale) {
    console.warn(
      `Log scale requires min > 0 and max > min. Got min=${min}, max=${max}. Falling back to linear scale.`
    );
  }

  const effectiveUseLogScale = canUseLogScale;
  const useStepCount = stepCount !== undefined && stepCount > 0;

  const valueToPosition = (val: number): number => {
    if (useStepCount) {
      if (effectiveUseLogScale) {
        return stepCount * Math.log(val / min) / Math.log(max / min);
      } else {
        return stepCount * (val - min) / (max - min);
      }
    }
    if (!effectiveUseLogScale) return val;
    return 100 * Math.log(val / min) / Math.log(max / min);
  };

  const positionToValue = (pos: number): number => {
    if (useStepCount) {
      if (effectiveUseLogScale) {
        return min * Math.pow(max / min, pos / stepCount);
      } else {
        return min + pos * (max - min) / stepCount;
      }
    }
    if (!effectiveUseLogScale) return pos;
    return min * Math.pow(max / min, pos / 100);
  };

  // Guard against undefined/NaN values
  const safeValue = typeof value === 'number' && !isNaN(value) ? value : min;
  const safeMin = typeof min === 'number' && !isNaN(min) ? min : 0;
  const safeMax = typeof max === 'number' && !isNaN(max) ? max : 100;

  const sliderMin = useStepCount ? 0 : (effectiveUseLogScale ? 0 : safeMin);
  const sliderMax = useStepCount ? stepCount : (effectiveUseLogScale ? 100 : safeMax);
  const sliderValue = (useStepCount || effectiveUseLogScale) ? valueToPosition(safeValue) : safeValue;
  const sliderStep = useStepCount ? 1 : (effectiveUseLogScale ? 0.1 : step);

  const rawPercentage = useStepCount
    ? (sliderValue / stepCount) * 100
    : (effectiveUseLogScale
      ? sliderValue
      : ((safeValue - safeMin) / (safeMax - safeMin)) * 100);

  const percentage = typeof rawPercentage === 'number' && !isNaN(rawPercentage)
    ? Number(rawPercentage.toFixed(6))
    : 0;

  const currentDisplayValue = formatValue ? formatValue(safeValue) : displayValue;

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const rawValue = parseFloat(e.target.value);
    const actualValue = (useStepCount || effectiveUseLogScale) ? positionToValue(rawValue) : rawValue;
    onChange(actualValue);
  };

  return (
    <div className="flex flex-col">
    <div className="relative">
      <input
        type="range"
        min={sliderMin}
        max={sliderMax}
        step={sliderStep}
        value={sliderValue}
        onChange={handleChange}
        onMouseDown={onMouseDown}
        onTouchStart={onTouchStart}
        onMouseUp={onMouseUp}
        onTouchEnd={onTouchEnd}
        onKeyDown={onKeyDown}
        onKeyUp={onKeyUp}
        onMouseEnter={onMouseEnter}
        onMouseLeave={onMouseLeave}
        onFocus={onFocus}
        onBlur={onBlur}
        className={className}
      />
      <div
          className="absolute top-full mt-1 text-[10px] text-black/60 font-medium transform -translate-x-1/2 pointer-events-none"
          style={{ left: `${percentage}%`, whiteSpace: 'nowrap'}}
      >
        {currentDisplayValue}
      </div>
      </div>
      {/* Spacer to reserve space for the absolutely positioned value label */}
      <div className="h-5" />
    </div>
  );
};
