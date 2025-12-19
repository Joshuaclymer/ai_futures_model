"use client";

import React from "react";

export default function TaskCompletionTimeChart(): React.JSX.Element {
  // Set dimensions and margins
  const width = 1110;
  const height = 915;
  const margin = { top: 60, right: 30, bottom: 270, left: 140 };
  const innerWidth = width - margin.left - margin.right;
  const innerHeight = height - margin.top - margin.bottom;

  // Define the y-axis scale (log scale for time)
  const timeValues = [
    "4 sec", "8 sec", "15 sec", "30 sec",
    "1 min", "2 min", "4 min", "8 min", "15 min", "30 min",
    "1 hr", "2 hrs", "4 hrs", "8 hrs", "16 hrs", "1 week", "2 weeks",
    "1 month", "2 months", "4 months", "8 months", "16 months", "32 months", "5 years"
  ];

  // Model point colors
  const gptcolor = "#003000"; // Dark green for OpenAI
  const claudecolor = "#832000"; // Dark orange for Anthropic
  // const grokcolor = "#808080"; // Gray for other companies (Grok, etc.)
  const backgroundcolor = "#505050"; // Dark gray for background points
  // const forecastcolor = "#40c040";

  // Trendline colors
  // const superexpColor = "#101090"; // Superexponential (intended) - solid
  // const superexpColorDashed = "#4040c0"; // Superexponential (intended) - dashed
  // const oldBuggyColor = "#999999"; // Original bugged superexponential - solid
  // const oldBuggyColorDashed = "#bbbbbb"; // Original bugged superexponential - dashed
  // New model colors (solid lines)
  const newMar2027Color = "#c00000"; // New model Mar 2027 - solid red
  const newMar2027ColorDashed = "#c04040"; // New model Mar 2027 - dashed
  const newJun2028Color = "#FF8C00"; // New model Jun 2028 - solid (dark orange)
  const newJun2028ColorDashed = "#FFB84D"; // New model Jun 2028 - dashed
  const newJun2030Color = "#9400D3"; // New model Jun 2030 - solid (dark violet)
  const newJun2030ColorDashed = "#B84DFF"; // New model Jun 2030 - dashed

  // Old model colors (dashed lines to distinguish)
  const oldMar2027Color = "#800000"; // Old model Mar 2027 - darker red
  const oldMar2027ColorDashed = "#a04040"; // Old model Mar 2027 - dashed
  const oldJun2028Color = "#CC7000"; // Old model Jun 2028 - darker orange
  const oldJun2028ColorDashed = "#CC9040"; // Old model Jun 2028 - dashed
  const oldJun2030Color = "#7000A8"; // Old model Jun 2030 - darker violet
  const oldJun2030ColorDashed = "#9040C0"; // Old model Jun 2030 - dashed
  // const metrExpColor = "#000000"; // METR Exponential - solid
  // const metrExpColorDashed = "#202020"; // METR Exponential - dashed

  // Define years for x-axis
  const years = ["2021", "2022", "2023", "2024", "2025", "2026", "2027", "2028"];

  // Model data points - including some from the original chart
  const modelData = [
    { model: "gpt-3.5-turbo-instruct", x: 2022.2, y: 1.45, color: gptcolor, shape: "triangle", showInLegend: true },
    { model: "GPT-4 0314", x: 2023.23, y: 4, color: gptcolor, shape: "square", showInLegend: true },
    { model: "GPT-4 1106", x: 2023.88, y: 4.55, color: backgroundcolor, shape: "diamond", showInLegend: false },
    // { model: "GPT-4o", x: 2024.3, y: 4.8, color: backgroundcolor, shape: "diamond2", showInLegend: false },
    { model: "o1-preview", x: 2024.65, y: 6.2, color: backgroundcolor, shape: "cross", showInLegend: false },
    { model: "o1", x: 2024.9, y: 6.7, color: gptcolor, shape: "plus", showInLegend: true },
    { model: "Claude 3.5 Sonnet (Old)", x: 2024.45, y: 5.65, color: claudecolor, shape: "triangle", showInLegend: true },
    { model: "Claude 3.5 Sonnet (New)", x: 2024.75, y: 6.2, color: claudecolor, shape: "square", showInLegend: true },
    { model: "Claude 3.7 Sonnet", x: 2025.15, y: 8, color: claudecolor, shape: "diamond2", showInLegend: true },
    // New METR points from 2025
    { model: "o3", x: 2025.3, y: 8.42, color: backgroundcolor, shape: "circle", showInLegend: false },
    // { model: "o4-mini", x: 2025.3, y: 8.00, color: backgroundcolor, shape: "plus", showInLegend: false },
    { model: "Claude Opus 4", x: 2025.375, y: 8.42, color: backgroundcolor, shape: "circle", showInLegend: false },
    // { model: "Claude Sonnet 4", x: 2025.375, y: 8.18, color: backgroundcolor, shape: "plus", showInLegend: false },
    // { model: "Grok 4", x: 2025.5, y: 8.00, color: backgroundcolor, shape: "diamond", showInLegend: false },
    { model: "Claude Opus 4.1", x: 2025.583, y: 8.49, color: backgroundcolor, shape: "diamond2", showInLegend: false },
    { model: "GPT-5", x: 2025.583, y: 8.79, color: gptcolor, shape: "diamond2", showInLegend: true },
    // { model: "Claude Sonnet 4.5", x: 2025.667, y: 8.42, color: backgroundcolor, shape: "cross", showInLegend: false },
    { model: "GPT-5.1-Codex-Max", x: 2025.833, y: 9.05, color: gptcolor, shape: "hexagon", showInLegend: true },
  ];

    // Y-axis values in minutes (corresponding to the labels)
    // Work time: 40 hours/week, so 1 work week = 40 hrs = 2400 min
    const timeValuesInMinutes = [
      4/60, 8/60, 15/60, 30/60, // seconds converted to minutes
      1, 2, 4, 8, 15, 30, // minutes
      60, 120, 240, 480, 960, // hours converted to minutes (1-16 hrs)
      2400, 4800, // 1 work week (40 hrs), 2 work weeks (80 hrs)
      9600, 19200, 38400, 76800, 153600, 307200, // 1-8 work months (160-1280 hrs, assuming 4 weeks/month)
      624000 // 5 work years (10,400 hrs = 52 weeks/year × 40 hrs/week × 5 years)
    ];

    // Helper function to convert old index values to minutes
    // For backward compatibility with existing model data points
    const indexToMinutes = (index: number): number => {
      if (index <= 0) return timeValuesInMinutes[0];
      if (index >= timeValuesInMinutes.length - 1) return timeValuesInMinutes[timeValuesInMinutes.length - 1];

      const lowerIndex = Math.floor(index);
      const upperIndex = Math.ceil(index);
      if (lowerIndex === upperIndex) return timeValuesInMinutes[lowerIndex];

      // Linear interpolation in log space
      const fraction = index - lowerIndex;
      const logLower = Math.log(timeValuesInMinutes[lowerIndex]);
      const logUpper = Math.log(timeValuesInMinutes[upperIndex]);
      const logValue = logLower + fraction * (logUpper - logLower);
      return Math.exp(logValue);
    };

    // Calculate scaling functions
    const xScale = (year: number): number => {
      const yearRange = [2021, 2028];
      return margin.left + ((year - yearRange[0]) / (yearRange[1] - yearRange[0])) * innerWidth;
    };

    // Maps minutes to pixel position using logarithmic scale
    const yScale = (minutes: number): number => {
      const minMinutes = timeValuesInMinutes[0];
      const maxMinutes = timeValuesInMinutes[timeValuesInMinutes.length - 1];
      const logMin = Math.log(minMinutes);
      const logMax = Math.log(maxMinutes);
      const logValue = Math.log(minutes);

      // Normalize to 0-1 range in log space
      const normalized = (logValue - logMin) / (logMax - logMin);

      // Convert to pixel position (inverted because canvas y increases downward)
      return margin.top + innerHeight - (normalized * innerHeight);
    };

    // Transition dates for each trendline (can be adjusted)
    const timelinesTransitionYear = 2025.25; // April 2025

    // Generate Timelines model path split at transition date
    const generateTimelinesModelPathSplit = async (transitionYear: number): Promise<{first: string, second: string}> => {
      try {
        const response = await fetch('/horizon_trajectories_sc_mar_2027_dist_incl_backcasts.csv');
        const csvText = await response.text();
        const lines = csvText.split('\n').slice(1);
        const maxWorkYears = timeValuesInMinutes[23]; // 5 work years in minutes

        let firstHalfPath = "";
        let secondHalfPath = "";
        let firstPointFirst = true;
        let firstPointSecond = true;
        let lastFirstHalfPoint: {x: number, y: number} | null = null;

        for (let i = 0; i < lines.length; i += 1) {
          const line = lines[i]?.trim();
          if (!line) continue;

          const columns = line.split(',');
          const calendarTime = Number(columns[0]); // year
          const timeHorizonMinutes = Number(columns[1]); // horizon_minutes
          if (isNaN(calendarTime) || isNaN(timeHorizonMinutes)) continue;
          if (calendarTime < 2021 || calendarTime > 2028) continue;

          // Skip points that are >= 5 work years
          if (timeHorizonMinutes >= maxWorkYears) continue;

          const x = xScale(calendarTime);
          const y = yScale(timeHorizonMinutes); // Use minutes directly
          if (isNaN(x) || isNaN(y)) continue;

          if (calendarTime <= transitionYear) {
            if (firstPointFirst) {
              firstHalfPath = `M ${x} ${y}`;
              firstPointFirst = false;
            } else {
              firstHalfPath += ` L ${x} ${y}`;
            }
            lastFirstHalfPoint = {x, y};
          } else {
            if (firstPointSecond) {
              if (lastFirstHalfPoint) {
                secondHalfPath = `M ${lastFirstHalfPoint.x} ${lastFirstHalfPoint.y} L ${x} ${y}`;
              } else {
                secondHalfPath = `M ${x} ${y}`;
              }
              firstPointSecond = false;
            } else {
              secondHalfPath += ` L ${x} ${y}`;
            }
          }
        }

        return {first: firstHalfPath, second: secondHalfPath};
      } catch (error) {
        console.error("Failed to load Timelines CSV:", error);
        return {first: "", second: ""};
      }
    };

    // Generate Daniel Median model path split at transition date
    const generateDanielModelPathSplit = async (transitionYear: number): Promise<{first: string, second: string}> => {
      try {
        const response = await fetch('/horizon_trajectories_sc_jun_2028_dist_incl_backcasts.csv');
        const csvText = await response.text();
        const lines = csvText.split('\n').slice(1);
        const maxWorkYears = timeValuesInMinutes[23]; // 5 work years in minutes

        let firstHalfPath = "";
        let secondHalfPath = "";
        let firstPointFirst = true;
        let firstPointSecond = true;
        let lastFirstHalfPoint: {x: number, y: number} | null = null;

        for (let i = 0; i < lines.length; i += 1) {
          const line = lines[i]?.trim();
          if (!line) continue;

          const columns = line.split(',');
          const calendarTime = Number(columns[0]); // year
          const timeHorizonMinutes = Number(columns[1]); // horizon_minutes
          if (isNaN(calendarTime) || isNaN(timeHorizonMinutes)) continue;
          if (calendarTime < 2021 || calendarTime > 2028) continue;

          // Skip points that are >= 5 work years
          if (timeHorizonMinutes >= maxWorkYears) continue;

          const x = xScale(calendarTime);
          const y = yScale(timeHorizonMinutes);
          if (isNaN(x) || isNaN(y)) continue;

          if (calendarTime <= transitionYear) {
            if (firstPointFirst) {
              firstHalfPath = `M ${x} ${y}`;
              firstPointFirst = false;
            } else {
              firstHalfPath += ` L ${x} ${y}`;
            }
            lastFirstHalfPoint = {x, y};
          } else {
            if (firstPointSecond) {
              if (lastFirstHalfPoint) {
                secondHalfPath = `M ${lastFirstHalfPoint.x} ${lastFirstHalfPoint.y} L ${x} ${y}`;
              } else {
                secondHalfPath = `M ${x} ${y}`;
              }
              firstPointSecond = false;
            } else {
              secondHalfPath += ` L ${x} ${y}`;
            }
          }
        }

        return {first: firstHalfPath, second: secondHalfPath};
      } catch (error) {
        console.error("Failed to load Daniel Median CSV:", error);
        return {first: "", second: ""};
      }
    };

    // Generate Eli Median model path split at transition date
    const generateEliModelPathSplit = async (transitionYear: number): Promise<{first: string, second: string}> => {
      try {
        const response = await fetch('/horizon_trajectories_sc_jun_2030_dist_incl_backcasts.csv');
        const csvText = await response.text();
        const lines = csvText.split('\n').slice(1);
        const maxWorkYears = timeValuesInMinutes[23]; // 5 work years in minutes

        let firstHalfPath = "";
        let secondHalfPath = "";
        let firstPointFirst = true;
        let firstPointSecond = true;
        let lastFirstHalfPoint: {x: number, y: number} | null = null;

        for (let i = 0; i < lines.length; i += 1) {
          const line = lines[i]?.trim();
          if (!line) continue;

          const columns = line.split(',');
          const calendarTime = Number(columns[0]); // year
          const timeHorizonMinutes = Number(columns[1]); // horizon_minutes
          if (isNaN(calendarTime) || isNaN(timeHorizonMinutes)) continue;
          if (calendarTime < 2021 || calendarTime > 2028) continue;

          // Skip points that are >= 5 work years
          if (timeHorizonMinutes >= maxWorkYears) continue;

          const x = xScale(calendarTime);
          const y = yScale(timeHorizonMinutes);
          if (isNaN(x) || isNaN(y)) continue;

          if (calendarTime <= transitionYear) {
            if (firstPointFirst) {
              firstHalfPath = `M ${x} ${y}`;
              firstPointFirst = false;
            } else {
              firstHalfPath += ` L ${x} ${y}`;
            }
            lastFirstHalfPoint = {x, y};
          } else {
            if (firstPointSecond) {
              if (lastFirstHalfPoint) {
                secondHalfPath = `M ${lastFirstHalfPoint.x} ${lastFirstHalfPoint.y} L ${x} ${y}`;
              } else {
                secondHalfPath = `M ${x} ${y}`;
              }
              firstPointSecond = false;
            } else {
              secondHalfPath += ` L ${x} ${y}`;
            }
          }
        }

        return {first: firstHalfPath, second: secondHalfPath};
      } catch (error) {
        console.error("Failed to load Eli Median CSV:", error);
        return {first: "", second: ""};
      }
    };

    // Generate OLD model Mar 2027 path from CSV
    const generateOldMar2027PathSplit = async (transitionYear: number): Promise<{first: string, second: string}> => {
      try {
        const response = await fetch('/old_mar_2027_central_25k_sims.csv');
        const csvText = await response.text();
        const lines = csvText.split('\n').slice(1);
        const maxWorkYears = timeValuesInMinutes[23];

        let firstHalfPath = "";
        let secondHalfPath = "";
        let firstPointFirst = true;
        let firstPointSecond = true;
        let lastFirstHalfPoint: {x: number, y: number} | null = null;

        for (let i = 0; i < lines.length; i += 1) {
          const line = lines[i]?.trim();
          if (!line) continue;

          const columns = line.split(',');
          const calendarTime = Number(columns[0]);
          const timeHorizonMinutes = Number(columns[2]); // central_incl_backcast_time_horizon_minutes
          if (isNaN(calendarTime) || isNaN(timeHorizonMinutes)) continue;
          if (calendarTime < 2021 || calendarTime > 2028) continue;
          if (timeHorizonMinutes >= maxWorkYears) continue;

          const x = xScale(calendarTime);
          const y = yScale(timeHorizonMinutes);
          if (isNaN(x) || isNaN(y)) continue;

          if (calendarTime <= transitionYear) {
            if (firstPointFirst) {
              firstHalfPath = `M ${x} ${y}`;
              firstPointFirst = false;
            } else {
              firstHalfPath += ` L ${x} ${y}`;
            }
            lastFirstHalfPoint = {x, y};
          } else {
            if (firstPointSecond) {
              if (lastFirstHalfPoint) {
                secondHalfPath = `M ${lastFirstHalfPoint.x} ${lastFirstHalfPoint.y} L ${x} ${y}`;
              } else {
                secondHalfPath = `M ${x} ${y}`;
              }
              firstPointSecond = false;
            } else {
              secondHalfPath += ` L ${x} ${y}`;
            }
          }
        }

        return {first: firstHalfPath, second: secondHalfPath};
      } catch (error) {
        console.error("Failed to load Old Mar 2027 CSV:", error);
        return {first: "", second: ""};
      }
    };

    // Generate OLD model Jun 2028 path from CSV
    const generateOldJun2028PathSplit = async (transitionYear: number): Promise<{first: string, second: string}> => {
      try {
        const response = await fetch('/old_jun_2028_central_25k_sims.csv');
        const csvText = await response.text();
        const lines = csvText.split('\n').slice(1);
        const maxWorkYears = timeValuesInMinutes[23];

        let firstHalfPath = "";
        let secondHalfPath = "";
        let firstPointFirst = true;
        let firstPointSecond = true;
        let lastFirstHalfPoint: {x: number, y: number} | null = null;

        for (let i = 0; i < lines.length; i += 1) {
          const line = lines[i]?.trim();
          if (!line) continue;

          const columns = line.split(',');
          const calendarTime = Number(columns[0]);
          const timeHorizonMinutes = Number(columns[2]); // central_incl_backcast_time_horizon_minutes
          if (isNaN(calendarTime) || isNaN(timeHorizonMinutes)) continue;
          if (calendarTime < 2021 || calendarTime > 2028) continue;
          if (timeHorizonMinutes >= maxWorkYears) continue;

          const x = xScale(calendarTime);
          const y = yScale(timeHorizonMinutes);
          if (isNaN(x) || isNaN(y)) continue;

          if (calendarTime <= transitionYear) {
            if (firstPointFirst) {
              firstHalfPath = `M ${x} ${y}`;
              firstPointFirst = false;
            } else {
              firstHalfPath += ` L ${x} ${y}`;
            }
            lastFirstHalfPoint = {x, y};
          } else {
            if (firstPointSecond) {
              if (lastFirstHalfPoint) {
                secondHalfPath = `M ${lastFirstHalfPoint.x} ${lastFirstHalfPoint.y} L ${x} ${y}`;
              } else {
                secondHalfPath = `M ${x} ${y}`;
              }
              firstPointSecond = false;
            } else {
              secondHalfPath += ` L ${x} ${y}`;
            }
          }
        }

        return {first: firstHalfPath, second: secondHalfPath};
      } catch (error) {
        console.error("Failed to load Old Jun 2028 CSV:", error);
        return {first: "", second: ""};
      }
    };

    // Generate OLD model Jun 2030 path from CSV
    const generateOldJun2030PathSplit = async (transitionYear: number): Promise<{first: string, second: string}> => {
      try {
        const response = await fetch('/old_jun_2030_central_25k_sims.csv');
        const csvText = await response.text();
        const lines = csvText.split('\n').slice(1);
        const maxWorkYears = timeValuesInMinutes[23];

        let firstHalfPath = "";
        let secondHalfPath = "";
        let firstPointFirst = true;
        let firstPointSecond = true;
        let lastFirstHalfPoint: {x: number, y: number} | null = null;

        for (let i = 0; i < lines.length; i += 1) {
          const line = lines[i]?.trim();
          if (!line) continue;

          const columns = line.split(',');
          const calendarTime = Number(columns[0]);
          const timeHorizonMinutes = Number(columns[2]); // central_incl_backcast_time_horizon_minutes
          if (isNaN(calendarTime) || isNaN(timeHorizonMinutes)) continue;
          if (calendarTime < 2021 || calendarTime > 2028) continue;
          if (timeHorizonMinutes >= maxWorkYears) continue;

          const x = xScale(calendarTime);
          const y = yScale(timeHorizonMinutes);
          if (isNaN(x) || isNaN(y)) continue;

          if (calendarTime <= transitionYear) {
            if (firstPointFirst) {
              firstHalfPath = `M ${x} ${y}`;
              firstPointFirst = false;
            } else {
              firstHalfPath += ` L ${x} ${y}`;
            }
            lastFirstHalfPoint = {x, y};
          } else {
            if (firstPointSecond) {
              if (lastFirstHalfPoint) {
                secondHalfPath = `M ${lastFirstHalfPoint.x} ${lastFirstHalfPoint.y} L ${x} ${y}`;
              } else {
                secondHalfPath = `M ${x} ${y}`;
              }
              firstPointSecond = false;
            } else {
              secondHalfPath += ` L ${x} ${y}`;
            }
          }
        }

        return {first: firstHalfPath, second: secondHalfPath};
      } catch (error) {
        console.error("Failed to load Old Jun 2030 CSV:", error);
        return {first: "", second: ""};
      }
    };

    // Use state to hold the model paths - NEW model
    const [newMar2027Paths, setNewMar2027Paths] = React.useState<{first: string, second: string}>({first: "", second: ""});
    const [newJun2028Paths, setNewJun2028Paths] = React.useState<{first: string, second: string}>({first: "", second: ""});
    const [newJun2030Paths, setNewJun2030Paths] = React.useState<{first: string, second: string}>({first: "", second: ""});

    // Use state to hold the model paths - OLD model
    const [oldMar2027Paths, setOldMar2027Paths] = React.useState<{first: string, second: string}>({first: "", second: ""});
    const [oldJun2028Paths, setOldJun2028Paths] = React.useState<{first: string, second: string}>({first: "", second: ""});
    const [oldJun2030Paths, setOldJun2030Paths] = React.useState<{first: string, second: string}>({first: "", second: ""});

    React.useEffect(() => {
      // Load NEW model trajectories
      generateTimelinesModelPathSplit(timelinesTransitionYear).then(setNewMar2027Paths);
      generateDanielModelPathSplit(timelinesTransitionYear).then(setNewJun2028Paths);
      generateEliModelPathSplit(timelinesTransitionYear).then(setNewJun2030Paths);
      // Load OLD model trajectories
      generateOldMar2027PathSplit(timelinesTransitionYear).then(setOldMar2027Paths);
      generateOldJun2028PathSplit(timelinesTransitionYear).then(setOldJun2028Paths);
      generateOldJun2030PathSplit(timelinesTransitionYear).then(setOldJun2030Paths);
    // eslint-disable-next-line react-hooks/exhaustive-deps
    }, []);

    // Draw shapes based on model type
    const drawModelPoint = (model: typeof modelData[0]) => {
      const x = xScale(model.x);
      const yMinutes = indexToMinutes(model.y); // Convert index to minutes
      const y = yScale(yMinutes);

      // Background points (gray) are small circles
      if (!model.showInLegend) {
        return (
          <circle
            cx={x}
            cy={y}
            r={6}
            fill={backgroundcolor}
            stroke="white"
            strokeWidth={1.5}
          />
        );
      }

      // Featured points are larger with their original shapes
      const size = 15;

      if (model.shape === "triangle") {
        return (
          <polygon
            points={`${x},${y-size/1.1} ${x-size/1.1},${y+size/1.1} ${x+size/1.1},${y+size/1.1}`}
            fill={model.color}
            stroke="white"
            strokeWidth={1.5}
          />
        );
      } else if (model.shape === "square") {
        return (
          <rect
            x={x - size/1.5}
            y={y - size/1.5}
            width={size*1.4}
            height={size*1.4}
            fill={model.color}
            stroke="white"
            strokeWidth={1.5}
          />
        );
      } else if (model.shape === "diamond") {
        return (
          <polygon
            points={`${x},${y-size} ${x+size},${y} ${x},${y+size} ${x-size},${y}`}
            fill={model.color}
            stroke="white"
            strokeWidth={1.5}
          />
        );
      } else if (model.shape === "diamond2") {
          return (
            <polygon
              points={`${x},${y-size} ${x+size/1.5},${y} ${x},${y+size} ${x-size/1.5},${y}`}
              fill={model.color}
              stroke="white"
              strokeWidth={1.5}
            />
          );
        } else if (model.shape === "plus") {
          return (
            <g>
              <line
                x1={x - size/1.0}
                y1={y}
                x2={x + size/1.0}
                y2={y}
                stroke="white"
                strokeWidth={9}
              />
              <line
                x1={x}
                y1={y - size/1.0}
                x2={x}
                y2={y + size/1.0}
                stroke="white"
                strokeWidth={9}
              />
              <line
                x1={x - size/1.0}
                y1={y}
                x2={x + size/1.0}
                y2={y}
                stroke={model.color}
                strokeWidth={6.5}
              />
              <line
                x1={x}
                y1={y - size/1.0}
                x2={x}
                y2={y + size/1.0}
                stroke={model.color}
                strokeWidth={6.5}
              />
            </g>
          );
        } else if (model.shape === "cross") {
          return (
            <g>
              <line
                x1={x - size/1.2}
                y1={y - size/1.2}
                x2={x + size/1.2}
                y2={y + size/1.2}
                stroke="white"
                strokeWidth={9}
              />
              <line
                x1={x - size/1.2}
                y1={y + size/1.2}
                x2={x + size/1.2}
                y2={y - size/1.2}
                stroke="white"
                strokeWidth={9}
              />
              <line
                x1={x - size/1.2}
                y1={y - size/1.2}
                x2={x + size/1.2}
                y2={y + size/1.2}
                stroke={model.color}
                strokeWidth={6.5}
              />
              <line
                x1={x - size/1.2}
                y1={y + size/1.2}
                x2={x + size/1.2}
                y2={y - size/1.2}
                stroke={model.color}
                strokeWidth={6.5}
              />
            </g>
          );
        } else if (model.shape === "hexagon") {
          // Hexagon shape
          const hexSize = size * 0.9;
          const angle = Math.PI / 3; // 60 degrees
          const points = [];
          for (let i = 0; i < 6; i++) {
            const px = x + hexSize * Math.cos(angle * i);
            const py = y + hexSize * Math.sin(angle * i);
            points.push(`${px},${py}`);
          }
          return (
            <polygon
              points={points.join(' ')}
              fill={model.color}
              stroke="white"
              strokeWidth={1.5}
            />
          );
        } else {
        return (
          <circle
            cx={x}
            cy={y}
            r={size/1.2}
            fill={model.color}
            stroke="white"
            strokeWidth={2}
          />
        );
      }
    };

    return (
      <div className="w-full relative overflow-hidden mt-8 mb-8">
        <svg
          width="100%"
          preserveAspectRatio="xMidYMid meet"
          viewBox={`0 0 ${width} ${height}`}
        >
          {/* Title */}
          <text
            x={xScale(2020.7)}
            y={margin.top - 35}
            textAnchor="start"
            fontSize="28"
            fontFamily="monospace"
            // fontWeight="bold"
            fill="#333333"
            className="text-[28px] font-mono"
          >
            Length Of Coding Tasks AI Agents Can Complete Autonomously
          </text>
          <text
            x={width / 2}
            y={margin.top / 2 +20}
            textAnchor="middle"
            fontSize="14"
            fontFamily="monospace"
            fill="#333333"
            className="text-[14px] font-mono"
          >
            80% success rate
          </text>

          {/* Y-axis label */}
          <text
            transform={`rotate(-90, ${margin.left / 3}, ${margin.top + innerHeight / 2})`}
            x={margin.left / 3}
            y={margin.top + innerHeight / 2.2}
            textAnchor="middle"
            fontSize="20"
            fontFamily="monospace"
            fill="#333333"
            className="text-[20px] font-mono"
          >
            Task time (for humans), 80% success rate
          </text>

          {/* X-axis label */}
          <text
            x={margin.left + innerWidth / 2}
            y={margin.top + innerHeight + 55}
            textAnchor="middle"
            fontSize="22"
            fontFamily="monospace"
            fill="#333333"
            className="text-[22px] font-mono"
          >
            AI Model Release Date
          </text>

          {/* Grid lines - horizontal */}
          {timeValues.map((_, i) => (
            <line
              key={`hgrid-${i}`}
              x1={margin.left}
              y1={yScale(timeValuesInMinutes[i])}
              x2={margin.left + innerWidth}
              y2={yScale(timeValuesInMinutes[i])}
              stroke="#CCCCCC"
              strokeWidth="1"
              opacity="0.5"
            />
          ))}

          {/* Grid lines - vertical */}
          {years.map((year) => (
            <line
              key={`vgrid-${year}`}
              x1={xScale(parseInt(year))}
              y1={margin.top}
              x2={xScale(parseInt(year))}
              y2={margin.top + innerHeight}
              stroke="#CCCCCC"
              strokeWidth="1"
              opacity="0.5"
            />
          ))}

          {/* Vertical line at 2025.25 (April 2025 - transition point) */}
          <line
            x1={xScale(2025.25)}
            y1={margin.top}
            x2={xScale(2025.25)}
            y2={margin.top + innerHeight}
            stroke="#8B0000"
            strokeWidth="1.5"
            strokeDasharray="2,3"
            opacity="0.6"
          />

          {/* Y-axis ticks and labels - reduced frequency */}
          {timeValues.map((time, i) => {
            const yPos = yScale(timeValuesInMinutes[i]);
            return (
            <g key={`ytick-${i}`}>
              <line
                x1={margin.left - 6+3*(1-i % 2)}
                y1={yPos}
                x2={margin.left}
                y2={yPos}
                stroke="#333333"
                strokeWidth="1.5"
              />
              <text
                x={margin.left - 10}
                y={(i % 2 === 1) ? yPos + 5: -10}
                textAnchor="end"
                fontSize="18"
                fontFamily="monospace"
                fill="#333333"
                className="text-[18px] font-mono"
              >
                {time}
              </text>
            </g>
            );
          })}

          {/* X-axis ticks and labels */}
          {years.map((year) => (
            <g key={`xtick-${year}`}>
              <line
                x1={xScale(parseInt(year))}
                y1={margin.top + innerHeight}
                x2={xScale(parseInt(year))}
                y2={margin.top + innerHeight + 5}
                stroke="#333333"
                strokeWidth="1.5"
              />
              <text
                x={xScale(parseInt(year))}
                y={margin.top + innerHeight + 25}
                textAnchor="middle"
                fontSize="18"
                fontFamily="monospace"
                fill="#333333"
                className="text-[18px] font-mono"
              >
                {year}
              </text>
            </g>
          ))}

          {/* NEW model Mar 2027 trajectory from CSV */}
          <path
          d={newMar2027Paths.first}
          fill="none"
          stroke={newMar2027Color}
          strokeWidth="3"
          opacity="0.8"
          />
          <path
          d={newMar2027Paths.second}
          fill="none"
          stroke={newMar2027ColorDashed}
          strokeWidth="3"
          strokeDasharray="7,7"
          opacity="0.8"
          />

          {/* NEW model Jun 2028 trajectory from CSV */}
          <path
          d={newJun2028Paths.first}
          fill="none"
          stroke={newJun2028Color}
          strokeWidth="3"
          opacity="0.8"
          />
          <path
          d={newJun2028Paths.second}
          fill="none"
          stroke={newJun2028ColorDashed}
          strokeWidth="3"
          strokeDasharray="7,7"
          opacity="0.8"
          />

          {/* NEW model Jun 2030 trajectory from CSV */}
          <path
          d={newJun2030Paths.first}
          fill="none"
          stroke={newJun2030Color}
          strokeWidth="3"
          opacity="0.8"
          />
          <path
          d={newJun2030Paths.second}
          fill="none"
          stroke={newJun2030ColorDashed}
          strokeWidth="3"
          strokeDasharray="7,7"
          opacity="0.8"
          />

          {/* OLD model Mar 2027 trajectory from CSV */}
          <path
          d={oldMar2027Paths.first}
          fill="none"
          stroke={oldMar2027Color}
          strokeWidth="2.5"
          opacity="0.7"
          strokeDasharray="3,3"
          />
          <path
          d={oldMar2027Paths.second}
          fill="none"
          stroke={oldMar2027ColorDashed}
          strokeWidth="2.5"
          strokeDasharray="7,7"
          opacity="0.7"
          />

          {/* OLD model Jun 2028 trajectory from CSV */}
          <path
          d={oldJun2028Paths.first}
          fill="none"
          stroke={oldJun2028Color}
          strokeWidth="2.5"
          opacity="0.7"
          strokeDasharray="3,3"
          />
          <path
          d={oldJun2028Paths.second}
          fill="none"
          stroke={oldJun2028ColorDashed}
          strokeWidth="2.5"
          strokeDasharray="7,7"
          opacity="0.7"
          />

          {/* OLD model Jun 2030 trajectory from CSV */}
          <path
          d={oldJun2030Paths.first}
          fill="none"
          stroke={oldJun2030Color}
          strokeWidth="2.5"
          opacity="0.7"
          strokeDasharray="3,3"
          />
          <path
          d={oldJun2030Paths.second}
          fill="none"
          stroke={oldJun2030ColorDashed}
          strokeWidth="2.5"
          strokeDasharray="7,7"
          opacity="0.8"
          />

          {/* Trendline info box */}
          <g transform={`translate(${margin.left + 15}, ${margin.top + 10})`}>
            <rect x="0" y="0" width="310" height="310" fill="#F5F5F5" rx="5" opacity="0.9" />

            <text
              x="155"
              y="22"
              fontSize="16"
              fontWeight="bold"
              fontFamily="monospace"
              textAnchor="middle"
              className="text-[16px] font-mono"
            >
              TRENDLINES
            </text>

            {/* NEW MODEL Section */}
            <text
              x="15"
              y="45"
              fontSize="12"
              fontWeight="bold"
              fontFamily="monospace"
              fill="#333333"
            >
              NEW Model (Dec 2025)
            </text>

            {/* New Mar 2027 */}
            <text x="25" y="65" fontSize="11" fontFamily="monospace" fill="#333333">
              SC Mar 2027
            </text>
            <line x1={150} y1={62} x2={190} y2={62} stroke={newMar2027Color} strokeWidth="3" opacity="0.8" />
            <line x1={190} y1={62} x2={230} y2={62} stroke={newMar2027ColorDashed} strokeWidth="3" strokeDasharray="7,7" opacity="0.8" />

            {/* New Jun 2028 */}
            <text x="25" y="85" fontSize="11" fontFamily="monospace" fill="#333333">
              SC Jun 2028 (Daniel median)
            </text>
            <line x1={220} y1={82} x2={255} y2={82} stroke={newJun2028Color} strokeWidth="3" opacity="0.8" />
            <line x1={255} y1={82} x2={295} y2={82} stroke={newJun2028ColorDashed} strokeWidth="3" strokeDasharray="7,7" opacity="0.8" />

            {/* New Jun 2030 */}
            <text x="25" y="105" fontSize="11" fontFamily="monospace" fill="#333333">
              SC Jun 2030 (Eli median)
            </text>
            <line x1={205} y1={102} x2={240} y2={102} stroke={newJun2030Color} strokeWidth="3" opacity="0.8" />
            <line x1={240} y1={102} x2={280} y2={102} stroke={newJun2030ColorDashed} strokeWidth="3" strokeDasharray="7,7" opacity="0.8" />

            {/* Divider */}
            <line x1={15} y1={120} x2={295} y2={120} stroke="#999999" strokeWidth="1" opacity="0.5" />

            {/* OLD MODEL Section */}
            <text
              x="15"
              y="140"
              fontSize="12"
              fontWeight="bold"
              fontFamily="monospace"
              fill="#333333"
            >
              OLD Model (Apr 2025)
            </text>

            {/* Old Mar 2027 */}
            <text x="25" y="160" fontSize="11" fontFamily="monospace" fill="#666666">
              SC Mar 2027
            </text>
            <line x1={150} y1={157} x2={190} y2={157} stroke={oldMar2027Color} strokeWidth="2.5" strokeDasharray="3,3" opacity="0.7" />
            <line x1={190} y1={157} x2={230} y2={157} stroke={oldMar2027ColorDashed} strokeWidth="2.5" strokeDasharray="7,7" opacity="0.7" />

            {/* Old Jun 2028 */}
            <text x="25" y="180" fontSize="11" fontFamily="monospace" fill="#666666">
              SC Jun 2028 (Daniel median)
            </text>
            <line x1={220} y1={177} x2={255} y2={177} stroke={oldJun2028Color} strokeWidth="2.5" strokeDasharray="3,3" opacity="0.7" />
            <line x1={255} y1={177} x2={295} y2={177} stroke={oldJun2028ColorDashed} strokeWidth="2.5" strokeDasharray="7,7" opacity="0.7" />

            {/* Old Jun 2030 */}
            <text x="25" y="200" fontSize="11" fontFamily="monospace" fill="#666666">
              SC Jun 2030 (Eli median)
            </text>
            <line x1={205} y1={197} x2={240} y2={197} stroke={oldJun2030Color} strokeWidth="2.5" strokeDasharray="3,3" opacity="0.7" />
            <line x1={240} y1={197} x2={280} y2={197} stroke={oldJun2030ColorDashed} strokeWidth="2.5" strokeDasharray="7,7" opacity="0.7" />

            {/* Divider */}
            <line x1={15} y1={215} x2={295} y2={215} stroke="#999999" strokeWidth="1" opacity="0.5" />

            {/* Legend explanation */}
            <text x="15" y="235" fontSize="10" fontFamily="monospace" fill="#666666">
              SC = Superhuman Coder milestone
            </text>
            <text x="15" y="250" fontSize="10" fontFamily="monospace" fill="#666666">
              Solid = historical, Dashed = forecast
            </text>
            <text x="15" y="268" fontSize="10" fontFamily="monospace" fill="#666666">
              New model: thicker solid lines
            </text>
            <text x="15" y="283" fontSize="10" fontFamily="monospace" fill="#666666">
              Old model: thinner dashed lines
            </text>
            <text x="15" y="300" fontSize="10" fontFamily="monospace" fill="#666666">
              (central trajectory incl. backcasts)
            </text>
          </g>
          <text
              x="715"
              y="497"
              fontSize="11"
              fontFamily="monospace"
              className="text-[11px] font-mono"
              fill="#666666"
            >
              Comparing OLD (Apr 2025) vs
            </text>
          <text
              x="715"
              y="509"
              fontSize="11"
              fontFamily="monospace"
              className="text-[11px] font-mono"
              fill="#666666"
            >
              NEW (Dec 2025) model trajectories.
            </text>
          <text
              x="140"
              y="720"
              fontSize="10"
              fontFamily="monospace"
              className="text-[10px] font-mono"
              fill="#666666"
            >
              This figure compares our OLD timelines model (Apr 2025, thinner dashed lines) with the NEW model (Dec 2025, thicker solid lines). Each set of three
            </text>
            <text
              x="140"
              y="733"
              fontSize="10"
              fontFamily="monospace"
              className="text-[10px] font-mono"
              fill="#666666"
            >
              curves represents central trajectories filtered for achieving superhuman coder (SC) in Mar 2027, Jun 2028, and Jun 2030 respectively.
            </text>
            <text
              x="140"
              y="746"
              fontSize="10"
              fontFamily="monospace"
              className="text-[10px] font-mono"
              fill="#666666"
            >
              A central trajectory has the least distance from the median horizon of all filtered trajectories, averaged across timesteps beginning in 2021 with
            </text>
            <text
              x="140"
              y="759"
              fontSize="10"
              fontFamily="monospace"
              className="text-[10px] font-mono"
              fill="#666666"
            >
              distance measured in log space. Red = Mar 2027, Orange = Jun 2028 (Daniel&apos;s median), Purple = Jun 2030 (Eli&apos;s median).
            </text>
            <text
              x="140"
              y="772"
              fontSize="10"
              fontFamily="monospace"
              className="text-[10px] font-mono"
              fill="#666666"
            >
              The NEW model uses an improved methodology. The OLD model central trajectories are shown for comparison.
            </text>
          {/* Legend - METR's DATA - Featured models only - Single column */}
          <g transform={`translate(${margin.left + 335}, ${margin.top + 10})`}>
            <rect x="0" y="0" width="250" height="250" fill="#F5F5F5" rx="5" opacity="0.8" />

            {/* Legend title */}
            <text
              x="125"
              y="25"
              textAnchor="middle"
              fontSize="18"
              fontWeight="bold"
              fontFamily="monospace"
            >
              METR&apos;s DATA
            </text>

            {/* OpenAI models */}
            <text x="15" y="50" fontSize="12" fontFamily="monospace" fontWeight="bold">OpenAI</text>

            <polygon points="21,60 15,71 27,71" fill={gptcolor} stroke="white" strokeWidth="1.5" />
            <text x="38" y="70" fontSize="11" fontFamily="monospace">gpt-3.5-turbo-inst</text>

            <rect x="15" y="83" width="12" height="12" fill={gptcolor} stroke="white" strokeWidth="1.5" />
            <text x="38" y="93" fontSize="11" fontFamily="monospace">GPT-4 0314</text>

            <g transform="translate(21, 110)">
              <line x1="-6" y1="0" x2="6" y2="0" stroke="white" strokeWidth="5" />
              <line x1="0" y1="-6" x2="0" y2="6" stroke="white" strokeWidth="5" />
              <line x1="-6" y1="0" x2="6" y2="0" stroke={gptcolor} strokeWidth="3.5" />
              <line x1="0" y1="-6" x2="0" y2="6" stroke={gptcolor} strokeWidth="3.5" />
            </g>
            <text x="38" y="114" fontSize="11" fontFamily="monospace">o1</text>

            {/* diamond2 for GPT-5 - taller to match graph */}
            <polygon points="21,124 26,131 21,138 16,131" fill={gptcolor} stroke="white" strokeWidth="1.5" />
            <text x="38" y="135" fontSize="11" fontFamily="monospace">GPT-5</text>

            {/* Hexagon for GPT-5.1-Codex-Max */}
            <polygon points="27,152 24,157 18,157 15,152 18,147 24,147" fill={gptcolor} stroke="white" strokeWidth="1.5" />
            <text x="38" y="155" fontSize="11" fontFamily="monospace">GPT-5.1-Codex-Max</text>

            {/* Anthropic models */}
            <text x="15" y="178" fontSize="12" fontFamily="monospace" fontWeight="bold">Anthropic</text>

            <polygon points="21,188 15,199 27,199" fill={claudecolor} stroke="white" strokeWidth="1.5" />
            <text x="38" y="198" fontSize="11" fontFamily="monospace">Claude 3.5 Sonnet (Old)</text>

            <rect x="15" y="210" width="12" height="12" fill={claudecolor} stroke="white" strokeWidth="1.5" />
            <text x="38" y="220" fontSize="11" fontFamily="monospace">Claude 3.5 Sonnet (New)</text>

            {/* diamond2 for Claude 3.7 Sonnet - taller to match graph */}
            <polygon points="21,230 26,237 21,244 16,237" fill={claudecolor} stroke="white" strokeWidth="1.5" />
            <text x="38" y="241" fontSize="11" fontFamily="monospace">Claude 3.7 Sonnet</text>

          </g>


          {/* Model data points - draw gray background points first, then colored points on top */}
          {modelData.filter(model => !model.showInLegend).map((model, i) => (
            <g key={`model-bg-${i}`}>
              {drawModelPoint(model)}
            </g>
          ))}
          {modelData.filter(model => model.showInLegend).map((model, i) => (
            <g key={`model-fg-${i}`}>
              {drawModelPoint(model)}
            </g>
          ))}

        </svg>
      </div>
    );
  }
