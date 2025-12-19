'use client';

import { useState, useCallback, useEffect } from 'react';
import type { SamplingConfig, ParameterDistribution } from '@/types/samplingConfig';
import {
  computeMedian,
  formatCI80,
  formatNumber,
  getDistributionTypeName,
} from '@/types/samplingConfig';
import { PARAMETER_RATIONALES } from '@/components/ParameterHoverContext';
import { DistributionPreviewChart } from '@/components/DistributionPreviewChart';
import { CorrelationMatrixHeatmap } from '@/components/CorrelationMatrixHeatmap';

interface SamplingDistributionsModalProps {
  isOpen: boolean;
  onClose: () => void;
  config: SamplingConfig;
  rawYaml: string;
  simulationId: string;
}

interface ParameterRowProps {
  name: string;
  distribution: ParameterDistribution;
}

function ParameterRow({ name, distribution }: ParameterRowProps) {
  const [showTooltip, setShowTooltip] = useState(false);
  const [showPreview, setShowPreview] = useState(false);

  const rationale = PARAMETER_RATIONALES[name];
  const median = computeMedian(distribution);
  const ci80Display = formatCI80(distribution.ci80);
  const medianDisplay = formatNumber(median);
  const distTypeName = getDistributionTypeName(distribution.dist);

  // Format display value for choice and fixed distributions
  let valueDisplay = '';
  if (distribution.dist === 'fixed') {
    valueDisplay = String(distribution.value);
  } else if (distribution.dist === 'choice') {
    const values = distribution.values ?? [];
    const probs = distribution.p ?? values.map(() => 1 / values.length);
    valueDisplay = values
      .map((v, i) => `${v} (${(probs[i] * 100).toFixed(0)}%)`)
      .join(', ');
  }

  return (
    <tr className="border-b border-gray-100 hover:bg-gray-50">
      {/* Parameter name with tooltip */}
      <td
        className="py-2 px-3 text-sm font-medium text-gray-900 relative"
        onMouseEnter={() => setShowTooltip(true)}
        onMouseLeave={() => setShowTooltip(false)}
      >
        <span className="cursor-help border-b border-dotted border-gray-400">
          {name}
        </span>
        {showTooltip && rationale && (
          <div className="absolute z-20 left-0 top-full mt-1 bg-gray-900 text-white text-xs px-3 py-2 rounded shadow-lg max-w-xs whitespace-normal">
            {rationale}
          </div>
        )}
      </td>

      {/* Distribution type */}
      <td className="py-2 px-3 text-sm text-gray-600">{distTypeName}</td>

      {/* Median */}
      <td className="py-2 px-3 text-sm text-gray-600 font-mono">
        {medianDisplay}
      </td>

      {/* CI80 with preview on hover */}
      <td
        className="py-2 px-3 text-sm text-gray-600 font-mono relative"
        onMouseEnter={() => setShowPreview(true)}
        onMouseLeave={() => setShowPreview(false)}
      >
        {distribution.dist === 'choice' || distribution.dist === 'fixed' ? (
          <span className="text-gray-500 italic text-xs">{valueDisplay}</span>
        ) : (
          <span className="cursor-help">{ci80Display}</span>
        )}
        {showPreview && distribution.dist !== 'fixed' && (
          <div className="absolute z-20 left-0 top-full mt-1 bg-white rounded shadow-lg border border-gray-200 p-2">
            <DistributionPreviewChart distribution={distribution} />
          </div>
        )}
      </td>
    </tr>
  );
}

export function SamplingDistributionsModal({
  isOpen,
  onClose,
  config,
  rawYaml,
  simulationId,
}: SamplingDistributionsModalProps) {
  const [activeTab, setActiveTab] = useState<'parameters' | 'timeseries' | 'correlation'>('parameters');

  // Close on escape key
  useEffect(() => {
    const handleEscape = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && isOpen) {
        onClose();
      }
    };
    window.addEventListener('keydown', handleEscape);
    return () => window.removeEventListener('keydown', handleEscape);
  }, [isOpen, onClose]);

  // Prevent body scroll when modal is open
  useEffect(() => {
    if (isOpen) {
      document.body.style.overflow = 'hidden';
    } else {
      document.body.style.overflow = '';
    }
    return () => {
      document.body.style.overflow = '';
    };
  }, [isOpen]);

  const handleDownload = useCallback(() => {
    const blob = new Blob([rawYaml], { type: 'application/x-yaml' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `sampling_config_${simulationId}.yaml`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  }, [rawYaml, simulationId]);

  const handleBackdropClick = useCallback(
    (e: React.MouseEvent) => {
      if (e.target === e.currentTarget) {
        onClose();
      }
    },
    [onClose]
  );

  if (!isOpen) return null;

  const parameters = config.parameters ?? {};
  const timeSeriesParameters = config.time_series_parameters ?? {};
  const hasCorrelation = !!config.correlation_matrix;
  const hasTimeSeries = Object.keys(timeSeriesParameters).length > 0;

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/50"
      onClick={handleBackdropClick}
    >
      <div className="bg-white rounded-lg shadow-xl max-w-4xl w-full mx-4 max-h-[90vh] flex flex-col">
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-gray-200">
          <div>
            <h2 className="text-lg font-semibold text-gray-900">
              Sampling Distributions
            </h2>
            <p className="text-sm text-gray-500 mt-0.5">
              Parameter distributions used for Monte Carlo sampling
            </p>
          </div>
          <button
            onClick={onClose}
            className="text-gray-400 hover:text-gray-600 transition-colors p-1"
            aria-label="Close"
          >
            <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>

        {/* Tabs */}
        <div className="flex border-b border-gray-200 px-6">
          <button
            onClick={() => setActiveTab('parameters')}
            className={`px-4 py-2 text-sm font-medium border-b-2 transition-colors ${
              activeTab === 'parameters'
                ? 'border-indigo-500 text-indigo-600'
                : 'border-transparent text-gray-500 hover:text-gray-700'
            }`}
          >
            Parameters ({Object.keys(parameters).length})
          </button>
          {hasTimeSeries && (
            <button
              onClick={() => setActiveTab('timeseries')}
              className={`px-4 py-2 text-sm font-medium border-b-2 transition-colors ${
                activeTab === 'timeseries'
                  ? 'border-indigo-500 text-indigo-600'
                  : 'border-transparent text-gray-500 hover:text-gray-700'
              }`}
            >
              Time Series ({Object.keys(timeSeriesParameters).length})
            </button>
          )}
          {hasCorrelation && (
            <button
              onClick={() => setActiveTab('correlation')}
              className={`px-4 py-2 text-sm font-medium border-b-2 transition-colors ${
                activeTab === 'correlation'
                  ? 'border-indigo-500 text-indigo-600'
                  : 'border-transparent text-gray-500 hover:text-gray-700'
              }`}
            >
              Correlations
            </button>
          )}
        </div>

        {/* Content */}
        <div className="flex-1 overflow-y-auto px-6 py-4">
          {activeTab === 'parameters' && (
            <table className="w-full">
              <thead>
                <tr className="text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  <th className="py-2 px-3">Parameter</th>
                  <th className="py-2 px-3">Type</th>
                  <th className="py-2 px-3">Median</th>
                  <th className="py-2 px-3">CI80 / Values</th>
                </tr>
              </thead>
              <tbody>
                {Object.entries(parameters).map(([name, dist]) => (
                  <ParameterRow key={name} name={name} distribution={dist} />
                ))}
              </tbody>
            </table>
          )}

          {activeTab === 'timeseries' && (
            <table className="w-full">
              <thead>
                <tr className="text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  <th className="py-2 px-3">Parameter</th>
                  <th className="py-2 px-3">Type</th>
                  <th className="py-2 px-3">Median</th>
                  <th className="py-2 px-3">CI80 / Values</th>
                </tr>
              </thead>
              <tbody>
                {Object.entries(timeSeriesParameters).map(([name, dist]) => (
                  <ParameterRow key={name} name={name} distribution={dist} />
                ))}
              </tbody>
            </table>
          )}

          {activeTab === 'correlation' && config.correlation_matrix && (
            <div className="space-y-4">
              <p className="text-sm text-gray-600">
                The correlation matrix below shows dependencies between parameters.
                Positive correlations (red) indicate parameters that tend to vary together,
                while negative correlations (blue) indicate inverse relationships.
              </p>
              <div className="overflow-x-auto">
                <CorrelationMatrixHeatmap correlationMatrix={config.correlation_matrix} />
              </div>
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="flex items-center justify-between px-6 py-4 border-t border-gray-200 bg-gray-50">
          <div className="text-sm text-gray-500">
            {config.num_samples && (
              <span>Samples: {config.num_samples.toLocaleString()}</span>
            )}
            {config.seed && (
              <span className="ml-4">Seed: {config.seed}</span>
            )}
          </div>
          <div className="flex gap-3">
            <button
              onClick={handleDownload}
              className="px-4 py-2 text-sm font-medium text-gray-700 bg-white border border-gray-300 rounded-md hover:bg-gray-50 transition-colors flex items-center gap-2"
            >
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 10v6m0 0l-3-3m3 3l3-3m2 8H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
              </svg>
              Download YAML
            </button>
            <button
              onClick={onClose}
              className="px-4 py-2 text-sm font-medium text-white bg-gray-800 rounded-md hover:bg-gray-700 transition-colors"
            >
              Close
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

export default SamplingDistributionsModal;
