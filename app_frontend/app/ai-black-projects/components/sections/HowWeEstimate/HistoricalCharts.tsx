'use client';

import { PlotlyChart } from '../../charts';
import { COLOR_PALETTE, rgba } from '../../colors';
import { CHART_MARGINS, CHART_HEIGHTS } from '../../chartConfig';

// ===== DETECTION LATENCY DATA =====
// This is actual data from the Flask app - Monte Carlo simulations and nuclear case studies
const DETECTION_LATENCY_DATA = {
  prediction: {
    x: [32, 33.9, 35.9, 38.1, 40.3, 42.7, 45.2, 47.9, 50.8, 53.8, 57.0, 60.4, 64.0, 67.8, 71.8, 76.1, 80.6, 85.4, 90.5, 95.8, 101.5, 107.6, 114.0, 120.7, 127.9, 135.5, 143.6, 152.1, 161.1, 170.7, 180.9, 191.6, 203.0, 215.1, 227.9, 241.4, 255.7, 271.0, 287.0, 304.1, 322.2, 341.3, 361.6, 383.1, 405.9, 430.0, 455.6, 482.6, 511.3, 541.7, 573.9, 608.0, 644.2, 682.4, 723.0, 765.9, 811.5, 859.7, 910.8, 964.9, 1022.3, 1083.1, 1147.4, 1215.6, 1287.9, 1364.4, 1445.5, 1531.4, 1622.5, 1718.9, 1821.0, 1929.3, 2043.9, 2165.4, 2294.1, 2430.5, 2574.9, 2727.9, 2890.1, 3061.9, 3243.8, 3436.6, 3640.9, 3857.3, 4086.5, 4329.4, 4586.7, 4859.3, 5148.1, 5454.1, 5778.3, 6121.7, 6485.5, 6870.9, 7279.4, 7712.0, 8170.4, 8655.9, 9170.4, 9715.5, 10000],
    y_median: [11.4, 11.1, 10.8, 10.5, 10.2, 9.9, 9.7, 9.4, 9.2, 8.9, 8.7, 8.5, 8.3, 8.1, 7.9, 7.7, 7.6, 7.4, 7.2, 7.1, 6.9, 6.8, 6.6, 6.5, 6.3, 6.2, 6.1, 6.0, 5.9, 5.8, 5.7, 5.6, 5.5, 5.4, 5.3, 5.2, 5.1, 5.0, 4.9, 4.9, 4.8, 4.7, 4.6, 4.6, 4.5, 4.4, 4.4, 4.3, 4.2, 4.2, 4.1, 4.0, 4.0, 3.9, 3.9, 3.8, 3.8, 3.7, 3.7, 3.6, 3.6, 3.5, 3.5, 3.4, 3.4, 3.3, 3.3, 3.2, 3.2, 3.1, 3.1, 3.1, 3.0, 3.0, 2.9, 2.9, 2.9, 2.8, 2.8, 2.8, 2.7, 2.7, 2.7, 2.6, 2.6, 2.6, 2.5, 2.5, 2.5, 2.4, 2.4, 2.4, 2.4, 2.3, 2.3, 2.3, 2.2, 2.2, 2.2, 2.1, 2.1],
    y_low: [1.9, 1.6, 1.7, 1.5, 1.5, 1.5, 1.4, 1.3, 1.2, 1.1, 1.1, 1.0, 1.0, 1.0, 0.9, 0.9, 0.8, 0.8, 0.7, 0.6, 0.6, 0.6, 0.5, 0.5, 0.5, 0.5, 0.4, 0.4, 0.4, 0.4, 0.4, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.08, 0.08, 0.07, 0.06, 0.05, 0.05, 0.04, 0.04, 0.04, 0.03, 0.03, 0.02, 0.02, 0.02, 0.02, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.008, 0.007, 0.006, 0.005, 0.004, 0.003, 0.003, 0.002, 0.002, 0.002, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001],
    y_high: [23.8, 23.6, 22.2, 21.8, 21.7, 20.6, 20.3, 20.0, 19.5, 19.1, 18.7, 18.4, 18.1, 17.8, 17.6, 17.3, 16.6, 16.3, 16.0, 15.7, 15.6, 15.2, 15.0, 14.6, 14.6, 14.1, 13.9, 13.5, 13.7, 13.6, 13.4, 13.1, 13.1, 13.0, 12.8, 12.8, 12.6, 12.6, 12.1, 12.4, 11.9, 11.8, 11.7, 11.9, 11.5, 11.7, 11.2, 11.0, 10.9, 10.8, 10.5, 10.9, 10.8, 10.7, 10.6, 10.4, 10.5, 10.3, 10.5, 9.9, 10.0, 10.3, 10.6, 9.6, 10.2, 10.4, 9.9, 10.2, 10.4, 9.5, 9.9, 10.1, 9.5, 9.9, 9.6, 10.0, 9.5, 9.8, 9.1, 9.3, 9.5, 9.1, 10.0, 9.0, 9.9, 9.0, 9.1, 9.1, 9.3, 8.8, 9.1, 9.2, 8.7, 8.8, 8.9, 9.0, 9.1, 8.4, 8.4, 8.2, 8.2],
  },
  actual_data: {
    x: [150, 115, 70, 60, 350, 40, 2000, 500, 3500, 200, 65, 1000],
    y: [1.0, 6.0, 7.0, 1.5, 1.0, 6.0, 3.0, 5.0, 4.0, 6.0, 9.0, 1.0],
    sites: [
      "Iran Fordow",
      "Syria Al-Kibar",
      "North Korea 2010",
      "Saudi (unconfirmed)",
      "Iran Natanz",
      "Libya",
      "Pakistan Kahuta",
      "North Korea 1980s",
      "Iraq PC-3",
      "South Africa",
      "Iran Lavizan-Shian",
      "Israel Dimona"
    ]
  }
};

// ===== INTELLIGENCE ACCURACY DATA =====
// Stated error bars data (from Flask app)
const INTELLIGENCE_ACCURACY_DATA = {
  statedErrorBars: [
    {category: "Nuclear Warheads", min: 150, max: 160, date: "1984"},
    {category: "Nuclear Warheads", min: 140, max: 157, date: "1999"},
    {category: "Nuclear Warheads", min: 225, max: 300, date: "1984"},
    {category: "Nuclear Warheads", min: 60, max: 80, date: "1999"},
    {category: "Fissile material (kg)", min: 25, max: 35, date: "1994"},
    {category: "Fissile material (kg)", min: 30, max: 50, date: "2007"},
    {category: "Fissile material (kg)", min: 17, max: 33, date: "1994"},
    {category: "Fissile material (kg)", min: 335, max: 400, date: "1998"},
    {category: "Fissile material (kg)", min: 330, max: 580, date: "1996"},
    {category: "Fissile material (kg)", min: 240, max: 395, date: "2000"},
    {category: "ICBM launchers", min: 10, max: 25, date: "1961"},
    {category: "ICBM launchers", min: 10, max: 25, date: "1961"},
    {category: "ICBM launchers", min: 105, max: 120, date: "1963"},
    {category: "ICBM launchers", min: 200, max: 240, date: "1964"},
    {category: "Intercontinental missiles", min: 180, max: 190, date: "2019"},
    {category: "Intercontinental missiles", min: 200, max: 300, date: "2025"},
    {category: "Intercontinental missiles", min: 192, max: 192, date: "2024"}
  ],
  estimatesVsReality: {
    estimates: [700, 800, 900, 300, 1000, 50, 800, 441, 18, 1000, 600, 428, 287, 311, 208],
    groundTruths: [610, 280, 847, 0, 1308, 60, 819, 499, 5, 1027, 661, 348, 308, 248, 287],
    categories: [
      "Aircraft", "Aircraft", "Aircraft",
      "Chemical Weapons", "Chemical Weapons",
      "Missiles", "Missiles", "Missiles", "Missiles",
      "Nuclear Warheads", "Nuclear Warheads",
      "Ground systems", "Ground systems", "Ground systems",
      "Troops"
    ],
    labels: [
      {index: 8, label: "Missile gap"},
      {index: 1, label: "Bomber gap"},
      {index: 3, label: "Iraq failure"}
    ]
  }
};

// ===== DETECTION LATENCY CHART =====
export function DetectionLatencyChart() {
  const data = DETECTION_LATENCY_DATA;
  const color = COLOR_PALETTE.detection;

  const traces: Plotly.Data[] = [
    // Confidence interval
    {
      x: data.prediction.x,
      y: data.prediction.y_high,
      mode: 'lines',
      line: { width: 0 },
      showlegend: false,
      hoverinfo: 'skip',
    },
    {
      x: data.prediction.x,
      y: data.prediction.y_low,
      mode: 'lines',
      line: { width: 0 },
      fill: 'tonexty',
      fillcolor: rgba('detection', 0.2),
      name: '90% CI    ',
      hoverinfo: 'skip',
    },
    // Median line
    {
      x: data.prediction.x,
      y: data.prediction.y_median,
      mode: 'lines',
      line: { color, width: 2 },
      name: 'Posterior Mean    ',
      hovertemplate: 'workers: %{x}<br>years: %{y:.1f}<extra></extra>'
    },
    // Actual data points
    {
      x: data.actual_data.x,
      y: data.actual_data.y,
      mode: 'markers+text' as Plotly.PlotData['mode'],
      marker: { size: 8, color, line: { color: 'white', width: 1 } },
      text: data.actual_data.sites,
      textposition: 'middle right' as Plotly.PlotData['textposition'],
      textfont: { size: 8 },
      name: 'Historical cases',
      hovertemplate: '%{text}<br>workers: %{x}<br>years: %{y:.1f}<extra></extra>',
      showlegend: false,
    },
  ];

  const layout: Partial<Plotly.Layout> = {
    xaxis: {
      title: { text: 'Nuclear-role workers', font: { size: 10 } },
      type: 'log',
      tickfont: { size: 9 },
      tickvals: [100, 1000, 10000],
      ticktext: ['100', '1,000', '10,000'],
      range: [Math.log10(20), Math.log10(10000)]
    },
    yaxis: {
      title: { text: 'Detection latency (years)', font: { size: 10 } },
      tickfont: { size: 9 },
      range: [0, 15],
    },
    showlegend: true,
    legend: {
      x: 0.02, y: 0.98,
      xanchor: 'left', yanchor: 'top',
      bgcolor: 'rgba(255,255,255,0.9)',
      font: { size: 8 }
    },
    margin: CHART_MARGINS.default,
    height: CHART_HEIGHTS.historical,
  };

  return <PlotlyChart data={traces} layout={layout} />;
}

// ===== INTELLIGENCE ACCURACY CHART =====
export function IntelligenceAccuracyChart() {
  const data = INTELLIGENCE_ACCURACY_DATA;
  const color = COLOR_PALETTE.datacenters_and_energy;

  // Calculate median error from estimates vs reality
  const estimateErrors = data.estimatesVsReality.estimates.map((est, i) => {
    const truth = data.estimatesVsReality.groundTruths[i];
    return truth !== 0 ? Math.abs((est - truth) / truth) * 100 : 0;
  }).filter(e => e > 0);
  const medianError = estimateErrors.sort((a,b) => a-b)[Math.floor(estimateErrors.length/2)];

  const maxVal = Math.max(...data.estimatesVsReality.estimates, ...data.estimatesVsReality.groundTruths);
  const xLine = Array.from({length: 50}, (_, i) => i * maxVal / 49);
  const xLineReversed = [...xLine].reverse(); // Create a copy before reversing
  const upperSlope = 1 + medianError / 100;
  const lowerSlope = 1 - medianError / 100;

  const traces: Plotly.Data[] = [
    // Error region - polygon going forward along upper bound, backward along lower bound
    {
      x: [...xLine, ...xLineReversed],
      y: [...xLine.map(x => upperSlope * x), ...xLineReversed.map(x => lowerSlope * x)],
      fill: 'toself',
      fillcolor: 'rgba(200,200,200,0.3)',
      line: { width: 0 },
      name: `Median error = ${medianError.toFixed(0)}%    `,
      hoverinfo: 'skip',
    },
    // y=x line
    {
      x: xLine,
      y: xLine,
      mode: 'lines',
      line: { color: 'grey', width: 1, dash: 'dash' },
      showlegend: false,
      hoverinfo: 'skip',
    },
    // Data points
    {
      x: data.estimatesVsReality.groundTruths,
      y: data.estimatesVsReality.estimates,
      mode: 'markers',
      marker: { color, size: 8 },
      name: 'Estimates',
      text: data.estimatesVsReality.categories,
      hovertemplate: '%{text}<br>Truth: %{x}<br>Estimate: %{y}<extra></extra>',
      showlegend: false,
    },
  ];

  const layout: Partial<Plotly.Layout> = {
    xaxis: {
      title: { text: 'Ground Truth', font: { size: 10 } },
      tickfont: { size: 9 },
    },
    yaxis: {
      title: { text: 'Estimate', font: { size: 10 } },
      tickfont: { size: 9 },
    },
    showlegend: true,
    legend: {
      x: 0.02, y: 0.98,
      xanchor: 'left', yanchor: 'top',
      bgcolor: 'rgba(255,255,255,0.9)',
      font: { size: 8 }
    },
    margin: CHART_MARGINS.default,
    height: CHART_HEIGHTS.historical,
    annotations: data.estimatesVsReality.labels.map(lbl => ({
      x: data.estimatesVsReality.groundTruths[lbl.index],
      y: data.estimatesVsReality.estimates[lbl.index],
      text: lbl.label,
      showarrow: true,
      arrowhead: 2,
      ax: 40,
      ay: -20,
      font: { size: 8 },
      bgcolor: '#ffffff',
    })),
  };

  return <PlotlyChart data={traces} layout={layout} />;
}
