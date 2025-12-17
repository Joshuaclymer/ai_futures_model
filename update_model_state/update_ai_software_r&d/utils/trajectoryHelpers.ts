import { ChartDataPoint } from '@/app/types';
import { formatTimeDuration } from './formatting';

export function mergeTrajectoryFields(
    trajectoryMap: { [year: number]: ChartDataPoint },
    point: ChartDataPoint & Record<string, unknown>,
    year: number
): void {
    if (!trajectoryMap[year]) {
        trajectoryMap[year] = {
            year,
            horizonLength: point.horizonLength,
            horizonFormatted: point.horizonFormatted || '',
            effectiveCompute: point.effectiveCompute
        };
    }

    // Merge trajectory fields from this individual trajectory
    Object.keys(point).forEach(key => {
        if (key.startsWith('trajectory_') || key.startsWith('effective_compute_trajectory_')) {
            trajectoryMap[year][key] = point[key];
        }
    });
}

export function processStaticTrajectoryData(
    trajectoryData: ChartDataPoint,
    trajectoryMap: { [year: number]: ChartDataPoint }
): void {
    if (Array.isArray(trajectoryData)) {
        // Nothing to process for array format - handled elsewhere
        return;
    }

    if (trajectoryData && typeof trajectoryData === 'object') {
        Object.keys(trajectoryData).forEach(key => {
            if (key.startsWith('trajectory_') || key.startsWith('effective_compute_trajectory_')) {
                const year = trajectoryData.year;
                mergeTrajectoryFields(trajectoryMap, trajectoryData, year);
            }
        });
    }
}

export function mergeDynamicTrajectoryData(
    data: ChartDataPoint[],
    trajectoryMap: { [year: number]: ChartDataPoint }
): void {
    data.forEach((point: ChartDataPoint) => {
        const year = point.year;
        mergeTrajectoryFields(trajectoryMap, point, year);
    });
}

export function finalizeTrajectoryMap(
    trajectoryMap: { [year: number]: ChartDataPoint }
): ChartDataPoint[] {
    return Object.values(trajectoryMap).sort((a, b) => a.year - b.year);
}
