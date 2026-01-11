import React, { createContext, useContext } from 'react';
import { ParametersType } from '@/constants/parameters';
import { ParameterSlider, ParameterSliderProps } from '../ui/ParameterSlider';
import { formatSCHorizon, formatAsPowerOfTenText, formatWorkTimeDuration } from '@/utils/formatting';
import { formatTo3SigFigs } from '@/utils/formatting';

interface ModelDefaults {
    'software_r_and_d.optimal_ces_eta_init'?: number;
    'software_r_and_d.automation_interp_type'?: string;
    'software_r_and_d.ai_research_taste_slope'?: number;
    anchor_progress_at_strong_cognitive_horizon?: number;
    present_year?: number;
    present_progress?: number;
    'software_r_and_d.progress_at_aa'?: number;
}

export interface ParameterConfig {
    defaults?: ModelDefaults;
    bounds?: Record<string, [number, number]>;
    metadata?: Record<string, unknown>;
}

export interface SimplificationCheckboxes {
    enableCodingAutomation: boolean;
    setEnableCodingAutomation: React.Dispatch<React.SetStateAction<boolean>>;
    enableExperimentAutomation: boolean;
    setEnableExperimentAutomation: React.Dispatch<React.SetStateAction<boolean>>;
    useExperimentThroughputCES: boolean;
    setUseExperimentThroughputCES: React.Dispatch<React.SetStateAction<boolean>>;
    enableSoftwareProgress: boolean;
    setEnableSoftwareProgress: React.Dispatch<React.SetStateAction<boolean>>;
    useComputeLaborGrowthSlowdown: boolean;
    setUseComputeLaborGrowthSlowdown: React.Dispatch<React.SetStateAction<boolean>>;
    useVariableHorizonDifficulty: boolean;
    setUseVariableHorizonDifficulty: React.Dispatch<React.SetStateAction<boolean>>;
}

export interface AdvancedSectionsProps {
    uiParameters: ParametersType;
    setUiParameters: React.Dispatch<React.SetStateAction<ParametersType>>;
    allParameters: ParameterConfig | null;
    isDragging: boolean;
    setIsDragging: React.Dispatch<React.SetStateAction<boolean>>;
    commitParameters: (nextParameters?: ParametersType) => void;
    scHorizonLogBounds: { min: number; max: number };
    preGapHorizonBounds: { min: number; max: number };
    parallelPenaltyBounds: { min: number; max: number };
    summary?: { beta_software?: number; r_software?: number;[key: string]: unknown } | null;
    lockedParameters?: Set<string>;
    onToggleTrajectoryDebugger?: () => void;
    isTrajectoryDebuggerOpen?: boolean;
    simplificationCheckboxes?: SimplificationCheckboxes;
    // Bounds from sampling config to ensure sliders stay within valid sampling ranges
    samplingConfigBounds?: Record<string, { min?: number; max?: number }>;
}

interface SliderContextValue {
    uiParameters: ParametersType;
    setUiParameters: React.Dispatch<React.SetStateAction<ParametersType>>;
    allParameters: ParameterConfig | null;
    isDragging: boolean;
    setIsDragging: React.Dispatch<React.SetStateAction<boolean>>;
    commitParameters: (nextParameters?: ParametersType) => void;
    lockedParameters?: Set<string>;
    samplingConfigBounds?: Record<string, { min?: number; max?: number }>;
}

const SliderContext = createContext<SliderContextValue | null>(null);

type SliderProps = Omit<ParameterSliderProps,
    'value' | 'uiParameters' | 'setUiParameters' | 'allParameters' |
    'isDragging' | 'setIsDragging' | 'commitParameters'
> & {
    paramName: string;
    lockedBy?: string | string[];
};

const Slider: React.FC<SliderProps> = ({ paramName, lockedBy, disabled, customMin, customMax, ...rest }) => {
    const ctx = useContext(SliderContext);
    if (!ctx) throw new Error('Slider must be used within SliderContext');

    const { uiParameters, setUiParameters, allParameters, isDragging, setIsDragging, commitParameters, lockedParameters, samplingConfigBounds } = ctx;

    const isLocked = (() => {
        if (!lockedParameters) return false;
        const names = Array.isArray(lockedBy) ? lockedBy : lockedBy ? [lockedBy] : [paramName];
        return names.some(n => lockedParameters.has(n));
    })();

    // Get bounds from sampling config if available, using provided customMin/Max as override
    const samplingBounds = samplingConfigBounds?.[paramName];
    const effectiveMin = customMin ?? samplingBounds?.min;
    const effectiveMax = customMax ?? samplingBounds?.max;

    return (
        <ParameterSlider
            paramName={paramName}
            value={uiParameters[paramName] as number}
            uiParameters={uiParameters}
            setUiParameters={setUiParameters}
            allParameters={allParameters}
            isDragging={isDragging}
            setIsDragging={setIsDragging}
            commitParameters={commitParameters}
            disabled={disabled || isLocked}
            customMin={effectiveMin}
            customMax={effectiveMax}
            {...rest}
        />
    );
};

const Toggle: React.FC<{
    label: string;
    description: string;
    checked: boolean;
    onChange: (checked: boolean) => void;
    disabled?: boolean;
}> = ({ label, description, checked, onChange, disabled }) => (
    <div className="mb-2 rounded-lg">
        <div className="flex items-center justify-between gap-2">
            <div className="min-w-0">
                <label className="block text-[11px] font-medium text-foreground leading-tight">{label}</label>
                <div className="text-[10px] text-gray-500 leading-tight">{description}</div>
            </div>
            <label className={`relative inline-flex items-center shrink-0 ${disabled ? 'cursor-not-allowed opacity-50' : 'cursor-pointer'}`}>
                <input
                    type="checkbox"
                    checked={checked}
                    onChange={(e) => onChange(e.target.checked)}
                    disabled={disabled}
                    className="sr-only peer"
                />
                <div className="w-8 h-4 bg-gray-200 peer-focus:outline-none peer-focus:ring-2 peer-focus:ring-blue-300 rounded-full peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-3 after:w-3 after:transition-all peer-checked:bg-blue-600 peer-disabled:opacity-50 peer-disabled:cursor-not-allowed" />
            </label>
        </div>
    </div>
);

const Section: React.FC<{
    title: string;
    subtitle?: string;
    children: React.ReactNode;
    className?: string;
}> = ({ title, subtitle, children, className = "mb-3" }) => (
    <details className={`${className} pt-1 rounded-none !m-0 !px-0 border-t border-b-0 border-l-0 border-r-0 border-gray-300`}>
        <summary className="text-[11px] font-bold cursor-pointer py-0.5">
            {title}
            {subtitle && <span className="text-[10px] text-gray-500 font-normal ml-1">{subtitle}</span>}
        </summary>
        <div className="mt-2">{children}</div>
    </details>
);

const SliderGrid: React.FC<{ children: React.ReactNode }> = ({ children }) => (
    <div className="grid grid-cols-1 gap-2">{children}</div>
);

const SimplificationCheckbox: React.FC<{
    id: string;
    label: string;
    tooltip: string;
    checked: boolean;
    onChange: (checked: boolean) => void;
}> = ({ id, label, tooltip, checked, onChange }) => (
    <div className="group relative flex items-center gap-1.5">
        <input
            type="checkbox"
            id={id}
            checked={checked}
            onChange={(e) => onChange(e.target.checked)}
            className="h-3 w-3 rounded border-gray-300 text-accent focus:ring-accent accent-accent cursor-pointer"
        />
        <label htmlFor={id} className="text-[10px] font-medium text-gray-900 cursor-pointer leading-tight">
            {label}
        </label>
        <div className="pointer-events-none absolute left-1/2 -translate-x-1/2 bottom-full mb-1 hidden w-48 rounded-md bg-gray-900 px-2 py-1.5 text-[10px] text-white shadow-lg group-hover:block z-50">
            <div className="relative">
                {tooltip}
                <div className="absolute left-1/2 -translate-x-1/2 top-full w-0 h-0 border-x-4 border-x-transparent border-t-4 border-t-gray-900"></div>
            </div>
        </div>
    </div>
);

export const AdvancedSections: React.FC<AdvancedSectionsProps> = ({
    uiParameters,
    setUiParameters,
    allParameters,
    isDragging,
    setIsDragging,
    commitParameters,
    scHorizonLogBounds,
    preGapHorizonBounds,
    parallelPenaltyBounds,
    summary,
    lockedParameters,
    onToggleTrajectoryDebugger,
    isTrajectoryDebuggerOpen,
    simplificationCheckboxes,
    samplingConfigBounds,
}) => {
    const handleToggle = (paramName: string) => (checked: boolean) => {
        const next = { ...uiParameters, [paramName]: checked };
        setUiParameters(next);
        commitParameters(next);
    };

    const handleTasteScheduleChange = (value: string) => {
        const next = { ...uiParameters, 'software_r_and_d.taste_schedule_type': value };
        setUiParameters(next);
        commitParameters(next);
    };

    const handleAutomationInterpTypeChange = (value: string) => {
        const next = { ...uiParameters, 'software_r_and_d.automation_interp_type': value };
        setUiParameters(next);
        commitParameters(next);
    };

    const ctxValue: SliderContextValue = {
        uiParameters, setUiParameters, allParameters, isDragging, setIsDragging, commitParameters, lockedParameters, samplingConfigBounds
    };

    return (
        <SliderContext.Provider value={ctxValue}>
            <div className="space-y-1 mt-4">
                {/* Inputs */}
                <details className="mb-3 pt-1 rounded-none !m-0 !px-0 border-t border-b-0 border-l-0 border-r-0 border-gray-300">
                    <summary className="text-[11px] font-bold cursor-pointer py-0.5">Inputs</summary>
                    <div className="mt-2">
                        <SliderGrid>
                            <Slider
                                paramName="compute.USComputeParameters.total_us_compute_annual_growth_rate"
                                label="US Frontier Project Compute Growth Rate"
                                description="Annual growth rate of US frontier project compute (OOMs/year)"
                                step={0.01}
                                fallbackMin={0.0}
                                fallbackMax={2.0}
                                decimalPlaces={2}
                                customFormatValue={(v) => `${v.toFixed(2)} OOMs/year`}
                            />
                        </SliderGrid>
                    </div>
                </details>

                {/* Time Horizon & Progress */}
                <Section
                    title="Coding Time Horizon Requirement"
                // subtitle={uiParameters['software_r_and_d.benchmarks_and_gaps_mode'] ? "(Benchmarks & Gaps Mode)" : "(Standard Mode)"}
                >
                    <div className={`mb-4 text-xs font-bold text-black/75 rounded-lg`}>
                        {uiParameters['software_r_and_d.benchmarks_and_gaps_mode']
                            ? 'Gap mode: AC is reached when the pre-gap horizon is met and after a further increase in effective compute equal to the specified gap.'
                            : 'Standard mode: AC is reached when the AC horizon threshold is met.'}
                    </div>

                    <Toggle
                        label="Include an Effective Compute Gap"
                        description="Require additional effective compute for AC to be achieved after the time horizon requirement is met."
                        checked={uiParameters['software_r_and_d.benchmarks_and_gaps_mode'] as boolean}
                        onChange={handleToggle('software_r_and_d.benchmarks_and_gaps_mode')}
                    />

                    <SliderGrid>
                        {uiParameters['software_r_and_d.benchmarks_and_gaps_mode'] ? (
                            <>
                                <Slider
                                    paramName="software_r_and_d.saturation_horizon_minutes"
                                    label="Pre-gap AC Horizon (Target)"
                                    description="Target horizon before adding the gap"
                                    customMin={preGapHorizonBounds.min}
                                    customMax={preGapHorizonBounds.max}
                                    customFormatValue={formatWorkTimeDuration}
                                    useLogScale
                                />
                                <Slider
                                    paramName="software_r_and_d.gap_years"
                                    label="Effective Compute Gap"
                                    description="A value of 1 means the magnitude of the gap is the effective compute increase in the present year."
                                    customMin={0.1}
                                    fallbackMax={10.0}
                                    step={0.1}
                                    customFormatValue={(v) => `${v.toFixed(1)} × 2025-effective-FLOP-growth`}
                                    useLogScale
                                />
                            </>
                        ) : (
                            <Slider
                                paramName="software_r_and_d.ac_time_horizon_minutes"
                                label="AC Time Horizon (Target)"
                                description="Target 80% reliability time horizon for Automated Coder determination"
                                customMin={scHorizonLogBounds.min}
                                customMax={scHorizonLogBounds.max}
                                step={0.1}
                                customFormatValue={formatSCHorizon}
                                useLogScale
                            />
                        )}
                    </SliderGrid>

                </Section>

                {/* Coding Automation */}
                <Section title="Coding Automation">
                    <SliderGrid>
                        <Slider
                            paramName="software_r_and_d.swe_multiplier_at_present_day"
                            label="Present Day Parallel Coding Labor Multiplier"
                            description="For what value of N would an AGI company in the present be indifferent between getting Nx more programmers and foregoing AI usage, vs. the status quo including AI usage."
                            step={0.05}
                            fallbackMin={1.0}
                            fallbackMax={10.0}
                            decimalPlaces={2}
                            useLogScale
                        />
                        <Slider
                            paramName="software_r_and_d.coding_automation_efficiency_slope"
                            label="Coding Automation Efficiency Slope (η)"
                            description="For a given task, each time we increase effective compute by the amount crossed in the present year (on top of the initial effective compute requirement to automate a task), by how many OOMs does the 'conversion rate' of GPUs->humans improve?"
                            step={0.1}
                            fallbackMin={0.01}
                            fallbackMax={10.0}
                            decimalPlaces={2}
                            customFormatValue={(v) => `${v.toFixed(2)} OOMs/effective-compute-OOMs`}
                            useLogScale
                        />
                        <Slider
                            paramName="software_r_and_d.rho_coding_labor"
                            label="Coding Labor Substitutability (ρ_c)"
                            description="This controls to what extent coding is a fixed series of tasks, vs. being able to substitute automated tasks for non-automated ones (lower values means more like the former)."
                            step={0.1}
                            fallbackMin={-10}
                            fallbackMax={0}
                            decimalPlaces={2}
                        />
                        <Slider
                            paramName="software_r_and_d.max_serial_coding_labor_multiplier"
                            label="Max Serial Coding Labor Multiplier"
                            description="At the physical limits of coding capability, AI could provide productivity benefits equivalent to speeding up all human coders by this much."
                            step={1.0}
                            stepCount={30}
                            fallbackMin={1.0}
                            fallbackMax={1e12}
                            customFormatValue={formatAsPowerOfTenText}
                            useLogScale
                        />
                        <div className="space-y-2">
                            <label className="block text-xs font-medium text-foreground">
                                Automation Schedule Type
                            </label>
                            <div className="text-xs text-gray-500 mb-2">
                                How automation fraction interpolates between anchor points
                            </div>
                            <select
                                value={uiParameters['software_r_and_d.automation_interp_type'] as string}
                                onChange={(e) => handleAutomationInterpTypeChange(e.target.value)}
                                className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500 text-sm"
                            >
                                <option value="linear">Linear</option>
                                <option value="logistic">Logistic</option>
                            </select>
                        </div>
                        {uiParameters['software_r_and_d.automation_interp_type'] === 'logistic' && (
                            <Slider
                                paramName="software_r_and_d.automation_logistic_asymptote"
                                label="Logistic Asymptote"
                                description="Upper asymptote for the logistic automation schedule. Values above 1 allow automation fraction to overshoot before being clipped to 1."
                                step={0.01}
                                fallbackMin={1.0001}
                                fallbackMax={2.0}
                                decimalPlaces={4}
                                useLogScale
                            />
                        )}
                    </SliderGrid>
                </Section>

                {/* CES Production Functions */}
                <Section
                    title="Experiment Throughput Production"
                // subtitle={uiParameters['software_r_and_d.direct_input_exp_cap_ces_params'] ? "(Direct Input Mode)" : "(Computed from Constraints)"}
                >
                    <div className="mb-4 text-xs font-bold">
                        Constraints mode: CES parameters are computed from constraints
                    </div>
                    <Toggle
                        label="Direct Input CES Params"
                        description="Use direct CES parameter input vs. computed from constraints"
                        checked={uiParameters['software_r_and_d.direct_input_exp_cap_ces_params'] as boolean}
                        onChange={handleToggle('software_r_and_d.direct_input_exp_cap_ces_params')}
                        disabled={lockedParameters?.has('software_r_and_d.direct_input_exp_cap_ces_params')}
                    />

                    {uiParameters['software_r_and_d.direct_input_exp_cap_ces_params'] ? (
                        <div className="space-y-4">
                            <div className="text-sm text-blue-600 bg-blue-50 p-3 rounded-lg">
                                Direct input mode: CES parameters are used as provided
                            </div>
                            <SliderGrid>
                                <Slider
                                    paramName="software_r_and_d.rho_experiment_capacity"
                                    label="Substitutability (ρ_x)"
                                    description="Controls the degree of substitutability between experiment compute and coding labor. Lower values mean more substitutable."
                                    step={0.01}
                                    fallbackMin={-1}
                                    fallbackMax={1}
                                    decimalPlaces={3}
                                />
                                <Slider
                                    paramName="software_r_and_d.alpha_experiment_capacity"
                                    label="Experiment Compute Weight (α)"
                                    description="Higher values mean experiment compute is more important relative to coding labor"
                                    step={0.01}
                                    fallbackMin={0.05}
                                    fallbackMax={0.95}
                                    decimalPlaces={3}
                                />
                                <Slider
                                    paramName="software_r_and_d.experiment_compute_exponent"
                                    label="Experiment Compute Discounting (ζ)"
                                    description="Experiment compute is taken to the exponent ζ before being combined with coding labor."
                                    step={0.01}
                                    fallbackMin={0.001}
                                    fallbackMax={10}
                                    decimalPlaces={3}
                                />
                                <Slider
                                    paramName="software_r_and_d.coding_labor_exponent"
                                    label="Coding Parallel Penalty (λ)"
                                    description="Multiplying the size of your coding labor force by N× is equivalent to speeding up coding labor by (N^λ)×. This is used to convert parallel to serial coding labor before entering the CES."
                                    customMin={parallelPenaltyBounds.min}
                                    customMax={parallelPenaltyBounds.max}
                                    step={0.01}
                                    decimalPlaces={3}
                                    lockedBy={['software_r_and_d.parallel_penalty', 'software_r_and_d.coding_labor_exponent']}
                                />
                            </SliderGrid>
                        </div>
                    ) : (
                        <div className="space-y-4">

                            <SliderGrid>
                                <Slider
                                    paramName="software_r_and_d.inf_labor_asymptote"
                                    label="Infinite Coding Labor Asymptote"
                                    description="By what factor faster AI software progress would go in 2024 if you immediately got unlimited coding labor."
                                    step={0.1}
                                    fallbackMin={1}
                                    fallbackMax={100000}
                                    decimalPlaces={1}
                                    useLogScale
                                />
                                <Slider
                                    paramName="software_r_and_d.inf_compute_asymptote"
                                    label="Infinite Experiment Compute Asymptote"
                                    description="By what factor faster AI software progress would go in 2024 if you immediately got unlimited experiment compute."
                                    step={10}
                                    fallbackMin={1}
                                    fallbackMax={100000}
                                    decimalPlaces={0}
                                    useLogScale
                                />
                                <Slider
                                    paramName="software_r_and_d.inv_compute_anchor_exp_cap"
                                    label="Slowdown from 10× less Experiment Compute"
                                    description="By what factor slower AI software progress would go in 2024 if you immediately had 10× less experiment compute."
                                    step={0.1}
                                    fallbackMin={1}
                                    fallbackMax={10}
                                    decimalPlaces={1}
                                    useLogScale
                                />
                                <Slider
                                    paramName="software_r_and_d.coding_labor_exponent"
                                    label="Coding Parallel Penalty (λ)"
                                    description="Multiplying the size of your coding labor force by N× is equivalent to speeding up coding labor by (N^λ)×. This is used to convert parallel to serial coding labor before entering the CES."
                                    customMin={parallelPenaltyBounds.min}
                                    customMax={parallelPenaltyBounds.max}
                                    step={0.01}
                                    decimalPlaces={3}
                                    lockedBy={['software_r_and_d.parallel_penalty', 'software_r_and_d.coding_labor_exponent']}
                                />
                            </SliderGrid>
                        </div>
                    )}
                </Section>

                {/* Experiment Selection Automation */}
                <Section title="Experiment Selection Automation">
                    <SliderGrid>
                        <Slider
                            paramName="software_r_and_d.ai_research_taste_at_coding_automation_anchor_sd"
                            label="AI Experiment Selection Skill at AC (SDs)"
                            description="When we reach SC, how good will AIs be at experiment selection relative to the median OpenBrain research scientist (0 SDs=median, ~3 SDs=best)"
                            step={0.1}
                            fallbackMin={-10}
                            fallbackMax={23}
                            decimalPlaces={1}
                        />
                        <Slider
                            paramName="software_r_and_d.ai_research_taste_slope"
                            label="AI Experiment Selection Slope (SDs/present-OOMs-per-year)"
                            description="For each amount of effective OOMs crossed in the present year, by how many SDs in the OpenBrain range is AIs' experiment selection increased?"
                            step={0.1}
                            fallbackMin={0.1}
                            fallbackMax={10.0}
                            decimalPlaces={1}
                            useLogScale
                        />
                        <Slider
                            paramName="software_r_and_d.median_to_top_taste_multiplier"
                            label="Median-to-Top-Human Experiment Selection Multiplier"
                            description="Ratio of the top researcher's experiment selection skill to the median researcher's skill"
                            step={0.1}
                            fallbackMin={1.1}
                            fallbackMax={20.0}
                            decimalPlaces={2}
                            useLogScale
                        />
                        <Slider
                            paramName="software_r_and_d.taste_limit"
                            label="Maximum Experiment Selection Skill"
                            description="Number of multiplicative median-to-top experiment selection gaps between the best humans and maximally capable AIs."
                            step={0.1}
                            fallbackMin={0}
                            fallbackMax={100}
                            decimalPlaces={1}
                            customFormatValue={(v) => `${v.toFixed(1)} top-to-median gaps above the best human`}
                        />
                        <Slider
                            paramName="software_r_and_d.taste_limit_smoothing"
                            label="Experiment Selection Slowdown Factor Halfway to Algorithmic Limit"
                            description="Halfway in log space to maximum experiment selection skill, each SD of experiment selection skill translates into the near-human-range-skill-per-SD taken to this power."
                            step={0.001}
                            fallbackMin={0.001}
                            fallbackMax={0.999}
                            decimalPlaces={3}
                        />
                    </SliderGrid>
                </Section>

                {/* General Capabilities */}
                <Section title="General Capabilities">
                    <SliderGrid>
                        <Slider
                            paramName="software_r_and_d.ted_ai_m2b"
                            label="Median-to-top-human jumps above SAR needed to reach TED-AI"
                            description="This multiplier on median-to-top experiment selection SDs above SAR corresponds to experiment selection skill at which TED-AI is achieved."
                            step={0.1}
                            fallbackMin={0}
                            fallbackMax={10}
                            decimalPlaces={1}
                            useLogScale
                        />
                    </SliderGrid>
                </Section>

                {/* Effective Compute */}
                <Section title="Effective Compute">
                    <SliderGrid>
                        <Slider
                            paramName="software_r_and_d.software_progress_rate_at_reference_year"
                            label="Software Efficiency OOMs/year in 2024"
                            description="In OOMs/year, how quickly was software efficiency growing in 2024?"
                            step={0.1}
                            fallbackMin={0.00001}
                            fallbackMax={10}
                            decimalPlaces={2}
                        />
                    </SliderGrid>

                    {summary?.beta_software != null && (
                        <div className="mt-4 text-xs text-black/75">
                            Calibrated Parameters: r={summary.r_software != null ? formatTo3SigFigs(summary.r_software) : '—'}, β={formatTo3SigFigs(summary.beta_software)} OOMs of research stock / OOM of software efficiency
                        </div>
                    )}
                </Section>

                {/* Extra Parameters */}
                <Section title="Extra (parameters we don't vary between simulations)">
                    <div className="">
                        <SliderGrid>
                            <Slider
                                paramName="software_r_and_d.present_day"
                                label="Present Day"
                                description="Used as a reference point for setting other parameters and for capability metrics."
                                step={0.1}
                                fallbackMin={2020.0}
                                fallbackMax={2030.0}
                                decimalPlaces={1}
                            />
                            <Slider
                                paramName="software_r_and_d.present_horizon"
                                label="Present Horizon"
                                description="The time horizon in present day; change if you change the present time."
                                step={0.1}
                                fallbackMin={0.01}
                                fallbackMax={100}
                                customFormatValue={formatWorkTimeDuration}
                                useLogScale
                            />
                            <Slider
                                paramName="software_r_and_d.automation_fraction_at_coding_automation_anchor"
                                label="Coding Automation Fraction at AC"
                                description="The fraction of coding tasks efficiently automatable at AC. This is constant at 1 during our simulations, only change if you understand the model."
                                step={0.01}
                                fallbackMin={0.01}
                                fallbackMax={1.0}
                                decimalPlaces={2}
                            />
                            <Slider
                                paramName="software_r_and_d.optimal_ces_eta_init"
                                label="Initial Automation Efficiency (η_init)"
                                description="The automation efficiency of a task at the point it is considered efficiently automated (via the coding automation fraction). Measured in coding FTEs per H100be."
                                step={0.001}
                                fallbackMin={1e-12}
                                fallbackMax={1e12}
                                decimalPlaces={4}
                                useLogScale
                            />
                            <Slider
                                paramName="software_r_and_d.top_percentile"
                                label="Percentile of Best Human Researcher"
                                description="The percentile of the best researcher in the AGI project human range"
                                step={0.001}
                                fallbackMin={0.5}
                                fallbackMax={0.99999}
                                decimalPlaces={5}
                                customFormatValue={(v) => `${(v * 100).toFixed(3)}%`}
                            />
                        </SliderGrid>
                    </div>
                </Section>

                {/* Configuration - Hidden */}
                <details className="mb-8 hidden">
                    <summary className="text-xs font-bold">Configuration</summary>
                    <div className="ml-4 mt-4">
                        <div className="grid grid-cols-1 gap-4">
                            <div className="space-y-2">
                                <label className="block text-sm font-medium text-foreground font-system-mono">
                                    Automation Interpolation Type
                                </label>
                                <div className="text-xs text-gray-500 mb-2">Type of automation interpolation</div>
                                <select
                                    value="linear"
                                    disabled
                                    className="w-full px-3 py-2 border border-gray-300 rounded-md bg-gray-100 text-gray-600 cursor-not-allowed"
                                >
                                    <option value="linear">Linear (model-supported)</option>
                                </select>
                                <p className="text-xs text-gray-500 mt-1">
                                    Note: only linear automation interpolation is implemented in the model today.
                                </p>
                            </div>

                            <div className="space-y-2">
                                <label className="block text-sm font-medium text-foreground font-system-mono">
                                    Taste Schedule Type
                                </label>
                                <div className="text-xs text-gray-500 mb-2">Type of research taste schedule</div>
                                <select
                                    value={uiParameters['software_r_and_d.taste_schedule_type'] as string}
                                    onChange={(e) => handleTasteScheduleChange(e.target.value)}
                                    className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
                                >
                                    <option value="SDs per effective OOM">SDs per effective OOM</option>
                                    <option value="SDs per progress-year">SDs per progress-year</option>
                                </select>
                            </div>
                        </div>
                    </div>
                </details>

                {/* Model Simplifications */}
                {simplificationCheckboxes && (
                    <Section title="Model Simplifications">
                        <SliderGrid>
                            <SimplificationCheckbox
                                id="enable-coding-automation"
                                label="Coding Automation"
                                tooltip="When disabled: Sets automation parameters to near-zero, effectively removing AI coding automation from the model."
                                checked={simplificationCheckboxes.enableCodingAutomation}
                                onChange={simplificationCheckboxes.setEnableCodingAutomation}
                            />
                            <SimplificationCheckbox
                                id="enable-experiment-automation"
                                label="Experiment Selection Automation"
                                tooltip="When disabled: Sets the median-to-top taste multiplier to near-one, removing the advantage of better experiment selection skill."
                                checked={simplificationCheckboxes.enableExperimentAutomation}
                                onChange={simplificationCheckboxes.setEnableExperimentAutomation}
                            />
                            <SimplificationCheckbox
                                id="enable-software-progress"
                                label="Software Progress"
                                tooltip="When disabled: Sets software efficiency growth rate to zero, removing algorithmic improvements over time."
                                checked={simplificationCheckboxes.enableSoftwareProgress}
                                onChange={simplificationCheckboxes.setEnableSoftwareProgress}
                            />
                            <SimplificationCheckbox
                                id="use-experiment-throughput-ces"
                                label="Experiment Throughput CES"
                                tooltip="When disabled: Uses Cobb-Douglas production function with very high asymptotes instead of the CES function."
                                checked={simplificationCheckboxes.useExperimentThroughputCES}
                                onChange={simplificationCheckboxes.setUseExperimentThroughputCES}
                            />
                            <SimplificationCheckbox
                                id="use-compute-labor-growth-slowdown"
                                label="Compute/Labor Growth Slowdown"
                                tooltip="When disabled: Uses constant growth rates instead of the specified slowdown behavior."
                                checked={simplificationCheckboxes.useComputeLaborGrowthSlowdown}
                                onChange={simplificationCheckboxes.setUseComputeLaborGrowthSlowdown}
                            />
                            <SimplificationCheckbox
                                id="use-variable-horizon-difficulty"
                                label="Variable Horizon Difficulty"
                                tooltip="When disabled: Sets doubling difficulty growth factor to 1, making horizon difficulty constant."
                                checked={simplificationCheckboxes.useVariableHorizonDifficulty}
                                onChange={simplificationCheckboxes.setUseVariableHorizonDifficulty}
                            />
                        </SliderGrid>
                    </Section>
                )}

                {/* Trajectory Debugger Toggle */}
                {onToggleTrajectoryDebugger && (
                    <div className="pt-4 mt-4 border-t border-gray-200">
                        <button
                            onClick={onToggleTrajectoryDebugger}
                            className={`text-xs font-mono ${isTrajectoryDebuggerOpen ? 'text-blue-600' : 'text-gray-500 hover:text-gray-700'}`}
                        >
                            {isTrajectoryDebuggerOpen ? '✓ Trajectory Debugger' : 'Trajectory Debugger'}
                        </button>
                        <span className="text-[10px] text-gray-400 ml-2">(Alt+D)</span>
                    </div>
                )}
            </div>
        </SliderContext.Provider>
    );
};
