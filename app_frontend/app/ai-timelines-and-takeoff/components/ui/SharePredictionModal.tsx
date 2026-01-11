'use client';

import { useState, useCallback, useEffect, useRef } from 'react';
import type { ChartDataPoint, BenchmarkPoint } from '@/app/types';
import type { MilestoneMap } from '@/types/milestones';
import { copyToClipboard, exportElementAsPng, EXPORT_PRESETS, EXPORT_COLORS } from '@/utils/chartExport';
import { CombinedChartExport } from '../export/CombinedChartExport';
import { StaticHorizonChart } from '../export/StaticHorizonChart';
import { StaticUpliftChart } from '../export/StaticUpliftChart';
import { DownloadIcon } from './DownloadIcon';

type ExportTab = 'combined' | 'horizon' | 'uplift';

export interface SharePredictionModalProps {
  isOpen: boolean;
  onClose: () => void;
  horizonChartData: ChartDataPoint[];
  upliftChartData: ChartDataPoint[];
  milestones: MilestoneMap | null;
  scHorizonMinutes: number;
  displayEndYear: number;
  shareUrl: string;
  benchmarkData?: BenchmarkPoint[];
}

const FONTS = {
  title: 'et-book, Georgia, serif',
  footer: 'Menlo, Consolas, monospace',
};

/**
 * Modal for sharing predictions with preview tabs, download, and copy link
 */
export function SharePredictionModal({
  isOpen,
  onClose,
  horizonChartData,
  upliftChartData,
  milestones,
  scHorizonMinutes,
  displayEndYear,
  shareUrl,
  benchmarkData = [],
}: SharePredictionModalProps) {
  const [activeTab, setActiveTab] = useState<ExportTab>('combined');
  const [isExporting, setIsExporting] = useState(false);
  const [copySuccess, setCopySuccess] = useState(false);
  const [showMETRData, setShowMETRData] = useState(false);
  const exportRef = useRef<HTMLDivElement>(null);

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

  const handleBackdropClick = useCallback(
    (e: React.MouseEvent) => {
      if (e.target === e.currentTarget) {
        onClose();
      }
    },
    [onClose]
  );

  const handleDownload = useCallback(async () => {
    if (!exportRef.current || isExporting) return;

    setIsExporting(true);
    try {
      const filename = activeTab === 'combined'
        ? 'ai-futures-trajectory'
        : activeTab === 'horizon'
        ? 'coding-time-horizon'
        : 'ai-software-rd-uplift';

      const preset = activeTab === 'combined'
        ? EXPORT_PRESETS.combined
        : EXPORT_PRESETS.individual;

      await exportElementAsPng(exportRef.current, filename, preset);
    } catch (error) {
      console.error('Failed to export:', error);
    } finally {
      setIsExporting(false);
    }
  }, [activeTab, isExporting]);

  const handleCopyLink = useCallback(async () => {
    const success = await copyToClipboard(shareUrl);
    if (success) {
      setCopySuccess(true);
      setTimeout(() => setCopySuccess(false), 2000);
    }
  }, [shareUrl]);

  if (!isOpen) return null;

  const tabs: { key: ExportTab; label: string }[] = [
    { key: 'combined', label: 'Combined' },
    { key: 'horizon', label: 'Horizon' },
    { key: 'uplift', label: 'Uplift' },
  ];

  // Get dimensions based on active tab
  const exportWidth = activeTab === 'combined' ? 1200 : 800;
  const exportHeight = activeTab === 'combined' ? 675 : 450;

  // Calculate preview scale to fit in modal
  const previewScale = activeTab === 'combined' ? 0.55 : 0.7;

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/50"
      onClick={handleBackdropClick}
    >
      <div className="bg-white rounded-lg shadow-xl max-w-5xl w-full mx-4 max-h-[90vh] flex flex-col">
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-gray-200">
          <div>
            <h2 className="text-lg font-semibold text-gray-900">
              Share Your Trajectory
            </h2>
            <p className="text-sm text-gray-500 mt-0.5">
              Download an image or copy a link to share
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

        {/* Tabs and Options */}
        <div className="flex items-center justify-between border-b border-gray-200 px-6">
          <div className="flex">
            {tabs.map(({ key, label }) => (
              <button
                key={key}
                onClick={() => setActiveTab(key)}
                className={`px-4 py-3 text-sm font-medium border-b-2 transition-colors ${
                  activeTab === key
                    ? 'text-gray-900 border-gray-900'
                    : 'text-gray-500 border-transparent hover:text-gray-700 hover:border-gray-300'
                }`}
              >
                {label}
              </button>
            ))}
          </div>
          {/* METR datapoints checkbox - only show for horizon-related tabs */}
          {(activeTab === 'combined' || activeTab === 'horizon') && benchmarkData.length > 0 && (
            <label className="flex items-center gap-2 text-sm text-gray-600 cursor-pointer select-none">
              <input
                type="checkbox"
                checked={showMETRData}
                onChange={(e) => setShowMETRData(e.target.checked)}
                className="w-4 h-4 rounded border-gray-300 text-gray-900 focus:ring-gray-500"
              />
              Show METR datapoints
            </label>
          )}
        </div>

        {/* Preview Area */}
        <div className="flex-1 overflow-auto p-6 bg-gray-50">
          <div className="flex justify-center">
            {/* Preview container with CSS scaling for display only */}
            <div
              className="bg-white shadow-lg rounded-lg overflow-hidden"
              style={{
                width: exportWidth * previewScale,
                height: exportHeight * previewScale,
              }}
            >
              {/* Export content at full size, scaled down via CSS transform */}
              <div
                ref={exportRef}
                style={{
                  width: exportWidth,
                  height: exportHeight,
                  transform: `scale(${previewScale})`,
                  transformOrigin: 'top left',
                }}
              >
                {activeTab === 'combined' && (
                  <CombinedChartExport
                    horizonChartData={horizonChartData}
                    upliftChartData={upliftChartData}
                    milestones={milestones}
                    scHorizonMinutes={scHorizonMinutes}
                    displayEndYear={displayEndYear}
                    benchmarkData={showMETRData ? benchmarkData : undefined}
                  />
                )}
                {activeTab === 'horizon' && (
                  <div
                    style={{
                      width: 800,
                      height: 450,
                      backgroundColor: EXPORT_COLORS.background,
                      display: 'flex',
                      flexDirection: 'column',
                      padding: '16px',
                      boxSizing: 'border-box',
                    }}
                  >
                    <h2
                      style={{
                        textAlign: 'center',
                        margin: 0,
                        marginBottom: '8px',
                        fontSize: 20,
                        fontWeight: 'bold',
                        fontFamily: FONTS.title,
                        color: EXPORT_COLORS.foreground,
                      }}
                    >
                      AI Futures Model - Coding Time Horizon (METR 80%)
                    </h2>
                    <div style={{ flex: 1 }}>
                      <StaticHorizonChart
                        chartData={horizonChartData}
                        scHorizonMinutes={scHorizonMinutes}
                        displayEndYear={displayEndYear}
                        width={768}
                        height={340}
                        title=""
                        benchmarkData={showMETRData ? benchmarkData : undefined}
                      />
                    </div>
                    <p
                      style={{
                        textAlign: 'center',
                        fontSize: 10,
                        fontFamily: FONTS.footer,
                        color: EXPORT_COLORS.foreground,
                        margin: 0,
                        lineHeight: 1.3,
                      }}
                    >
                      The coding time horizon is the maximum length of coding tasks frontier AI systems can complete with a success rate of 80%, with the length defined as the time taken by typical AI company employees who do similar tasks.
                    </p>
                    <p
                      style={{
                        textAlign: 'center',
                        fontSize: 11,
                        fontFamily: FONTS.footer,
                        color: EXPORT_COLORS.foreground,
                        margin: 0,
                        marginTop: '8px',
                      }}
                    >
                      aifuturesmodel.com
                    </p>
                  </div>
                )}
                {activeTab === 'uplift' && (
                  <div
                    style={{
                      width: 800,
                      height: 450,
                      backgroundColor: EXPORT_COLORS.background,
                      display: 'flex',
                      flexDirection: 'column',
                      padding: '16px',
                      boxSizing: 'border-box',
                    }}
                  >
                    <h2
                      style={{
                        textAlign: 'center',
                        margin: 0,
                        marginBottom: '8px',
                        fontSize: 20,
                        fontWeight: 'bold',
                        fontFamily: FONTS.title,
                        color: EXPORT_COLORS.foreground,
                      }}
                    >
                      AI Futures Model - AI Software R&D Uplift
                    </h2>
                    <div style={{ flex: 1 }}>
                      <StaticUpliftChart
                        chartData={upliftChartData}
                        milestones={milestones}
                        displayEndYear={displayEndYear}
                        width={768}
                        height={340}
                        title=""
                      />
                    </div>
                    <p
                      style={{
                        textAlign: 'center',
                        fontSize: 10,
                        fontFamily: FONTS.footer,
                        color: EXPORT_COLORS.foreground,
                        margin: 0,
                        lineHeight: 1.3,
                      }}
                    >
                      The AI Software R&D Uplift is the speedup in software progress that would be achieved if the frontier AI systems at a given time were deployed within today's leading AI company. (In previous work we called this the AI R&D progress multiplier.)
                    </p>
                    <p
                      style={{
                        textAlign: 'center',
                        fontSize: 11,
                        fontFamily: FONTS.footer,
                        color: EXPORT_COLORS.foreground,
                        margin: 0,
                        marginTop: '8px',
                      }}
                    >
                      aifuturesmodel.com
                    </p>
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>

        {/* Actions */}
        <div className="flex items-center justify-between px-6 py-4 border-t border-gray-200 bg-white">
          <div className="text-sm text-gray-500">
            {activeTab === 'combined' ? '1200 x 675 px' : '800 x 450 px'}
          </div>
          <div className="flex gap-3">
            <button
              onClick={handleCopyLink}
              className={`px-4 py-2 text-sm font-medium rounded-md border transition-colors ${
                copySuccess
                  ? 'bg-green-50 text-green-700 border-green-300'
                  : 'bg-white text-gray-700 border-gray-300 hover:bg-gray-50'
              }`}
            >
              {copySuccess ? 'Copied!' : 'Copy Link'}
            </button>
            <button
              onClick={handleDownload}
              disabled={isExporting}
              className="px-4 py-2 text-sm font-medium text-white bg-gray-900 rounded-md hover:bg-gray-800 transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
            >
              {isExporting ? (
                <>
                  <span className="h-4 w-4 animate-spin rounded-full border-2 border-white border-t-transparent" />
                  Exporting...
                </>
              ) : (
                <>
                  <DownloadIcon size={16} />
                  Download PNG
                </>
              )}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
