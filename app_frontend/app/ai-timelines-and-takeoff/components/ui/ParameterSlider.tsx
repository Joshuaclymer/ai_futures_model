import React, { useState, useId, useRef, useEffect } from 'react';
import { SliderWithThumbValue } from './SliderWithThumbValue';
import { ParametersType } from '@/constants/parameters';
import { SliderPopover } from './SliderPopover';
import { ChartConfig } from '@/types/chartConfig';
import { ParameterConfig } from '../sections/AdvancedSections';
import { useParameterHover, PARAMETER_RATIONALES } from '@/components/ParameterHoverContext';
import { PARAMETER_BOUNDS_OVERRIDES } from '@/constants/parameterBoundsOverrides';

const SLIDER_KEYBOARD_KEYS = new Set([
    'ArrowLeft',
    'ArrowRight',
    'ArrowUp',
    'ArrowDown',
    'Home',
    'End',
    'PageUp',
    'PageDown',
]);


export interface ParameterSliderProps {
    // Core props
    paramName: keyof ParametersType & string;
    label: string;
    description?: string;

    // Slider configuration
    step?: number | string;
    fallbackMin?: number;
    fallbackMax?: number;

    // Override bounds/step (for dynamic values)
    customMin?: number;
    customMax?: number;
    customStep?: number | string;

    // Formatting
    decimalPlaces?: number;
    customFormatValue?: (value: number) => string;

    // State and handlers
    value: number;
    uiParameters: ParametersType;
    setUiParameters: React.Dispatch<React.SetStateAction<ParametersType>>;
    allParameters: ParameterConfig | null;
    isDragging: boolean;
    setIsDragging: React.Dispatch<React.SetStateAction<boolean>>;
    commitParameters: (nextParameters?: ParametersType) => void;

    // Optional event handlers for popovers
    onMouseEnter?: () => void;
    onMouseLeave?: () => void;
    onFocus?: () => void;
    onBlur?: () => void;

    // Popover support
    popoverConfig?: ChartConfig;
    popoverVisible?: boolean;
    onPopoverClose?: () => void;
    horizonParams?: {
        uses_shifted_form: boolean;
        anchor_progress: number | null;
    } | null;

    // Optional styling
    className?: string;
    
    // Disabled state
    disabled?: boolean;
    
    // Log scale
    useLogScale?: boolean;
    
    // Step count - divide range into exactly N steps
    stepCount?: number;
}

export const ParameterSlider: React.FC<ParameterSliderProps> = ({
    paramName,
    label,
    description,
    step = 0.1,
    fallbackMin = 0,
    fallbackMax = 100,
    customMin,
    customMax,
    customStep,
    decimalPlaces = 2,
    customFormatValue,
    value,
    uiParameters,
    setUiParameters,
    allParameters,
    setIsDragging,
    commitParameters,
    onMouseEnter,
    onMouseLeave,
    onFocus,
    onBlur,
    popoverConfig,
    popoverVisible,
    onPopoverClose,
    horizonParams,
    className,
    disabled = false,
    useLogScale = false,
    stepCount,
}) => {
    const { setHoveredParameter } = useParameterHover();
    const [showRationale, setShowRationale] = useState(false);
    const [tooltipPosition, setTooltipPosition] = useState<{ top: number; left: number } | null>(null);
    const containerRef = useRef<HTMLDivElement>(null);
    const anchorId = useId();
    const anchorName = `--param-slider-${anchorId.replace(/:/g, '')}`;

    // Get bounds override if available
    const boundsOverride = PARAMETER_BOUNDS_OVERRIDES[paramName];
    
    // Priority: override > customMin/Max > API bounds > fallback
    const min = boundsOverride?.min ?? customMin ?? (allParameters?.bounds?.[paramName as string]?.[0] ?? fallbackMin);
    const max = boundsOverride?.max ?? customMax ?? (allParameters?.bounds?.[paramName as string]?.[1] ?? fallbackMax);
    const actualStep = customStep ?? step;
    const formatValue = customFormatValue || ((val: number) => val.toFixed(decimalPlaces));
    const handleChange = (newValue: number) => {
        setUiParameters(prev => ({ ...prev, [paramName]: newValue }));
    };
    const displayMin = boundsOverride?.min ?? customMin ?? (allParameters?.bounds?.[paramName as string]?.[0] ?? fallbackMin);
    const displayMax = boundsOverride?.max ?? customMax ?? (allParameters?.bounds?.[paramName as string]?.[1] ?? fallbackMax);
    const fullDescription = description ? `${description} (${displayMin} to ${displayMax})` : undefined;
    const wrapperClass = popoverConfig ? "space-y-0.5 relative" : `${className || "space-y-0.5"} relative`;
    const containerClass = disabled ? `${wrapperClass} opacity-50 pointer-events-none` : wrapperClass;
    const rationale = PARAMETER_RATIONALES[paramName];

    // Calculate tooltip position as a fallback for browsers that don't support CSS anchor positioning
    useEffect(() => {
        if (showRationale && containerRef.current) {
            const rect = containerRef.current.getBoundingClientRect();
            const tooltipHeight = 100; // Approximate tooltip height
            const viewportHeight = window.innerHeight;

            // Position below the slider by default, but above if there's not enough space
            const spaceBelow = viewportHeight - rect.bottom;
            const top = spaceBelow > tooltipHeight
                ? rect.bottom + 4
                : rect.top - tooltipHeight - 4;

            setTooltipPosition({
                top,
                left: rect.left,
            });
        }
    }, [showRationale]);

    const handleMouseEnter = () => {
        setHoveredParameter(paramName);
        setShowRationale(true);
        onMouseEnter?.();
    };
    
    const handleMouseLeave = () => {
        setHoveredParameter(null);
        setShowRationale(false);
        onMouseLeave?.();
    };
    
    const handleFocus = () => {
        setHoveredParameter(paramName);
        setShowRationale(true);
        onFocus?.();
    };
    
    const handleBlur = () => {
        setHoveredParameter(null);
        setShowRationale(false);
        onBlur?.();
    };

    const handleKeyDown = (event: React.KeyboardEvent<HTMLInputElement>) => {
        if (disabled) {
            return;
        }
        if (SLIDER_KEYBOARD_KEYS.has(event.key)) {
            setIsDragging(true);
        }
    };

    const handleKeyUp = (event: React.KeyboardEvent<HTMLInputElement>) => {
        if (disabled) {
            return;
        }
        if (SLIDER_KEYBOARD_KEYS.has(event.key)) {
            commitParameters();
        }
    };

    const showCustomChartPopover = popoverConfig && popoverVisible && onPopoverClose;
    
    return (
        <div
            ref={containerRef}
            className={containerClass}
            onMouseEnter={handleMouseEnter}
            onMouseLeave={handleMouseLeave}
            onFocus={handleFocus}
            onBlur={handleBlur}
            style={{ anchorName }}
        >
            {label && (
                <label className="block text-[11px] font-medium text-foreground font-sans leading-tight">
                    {label}
                    {disabled && <span className="ml-1 text-[10px] text-gray-400">(Locked)</span>}
                </label>
            )}
            <SliderWithThumbValue
                value={value}
                min={min}
                max={max}
                step={actualStep}
                onChange={handleChange}
                onMouseDown={() => !disabled && setIsDragging(true)}
                onMouseUp={() => !disabled && commitParameters()}
                onTouchStart={() => !disabled && setIsDragging(true)}
                onTouchEnd={() => !disabled && commitParameters()}
                onKeyDown={handleKeyDown}
                onKeyUp={handleKeyUp}
                displayValue=""
                formatValue={formatValue}
                useLogScale={useLogScale}
                stepCount={stepCount}
            />
            {/* Render SliderPopover conditionally with config */}
            {showCustomChartPopover && (
                <SliderPopover
                    config={popoverConfig}
                    visible={popoverVisible}
                    onClose={onPopoverClose}
                    uiParameters={uiParameters}
                    horizonParams={horizonParams}
                    anchorName={anchorName}
                />
            )}
            {/* Render rationale tooltip on hover - uses CSS anchor positioning with JS fallback */}
            {showRationale && rationale && tooltipPosition && (
                <div
                    className="fixed z-[9999] p-3 bg-gray-900 text-white text-xs rounded-lg shadow-lg max-w-xs font-system-mono pointer-events-none"
                    style={{
                        top: `${tooltipPosition.top}px`,
                        left: `${tooltipPosition.left}px`,
                    }}
                >
                    <div className="font-semibold mb-1">Default Value Rationale:</div>
                    <div className="text-gray-200">{rationale}</div>
                </div>
            )}
        </div>
    );
};
