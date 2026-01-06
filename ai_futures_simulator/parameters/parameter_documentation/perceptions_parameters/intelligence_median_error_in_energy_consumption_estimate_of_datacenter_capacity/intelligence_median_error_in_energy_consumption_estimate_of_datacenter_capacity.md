# Energy Accounting Detection

## Purpose

Models the probability that US intelligence detects covert datacenters by identifying discrepancies between China's total energy consumption and accounted-for energy usage.

## Parameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| Intelligence median error (energy) | 7% | 1–50% | Median percentage error in US estimates of PRC datacenter energy consumption |

## How Detection Works

Detection probability depends on two factors:

**1. Covert energy consumption**

Covert datacenters consume significant energy. The larger the covert datacenter capacity, the larger the energy discrepancy that must be hidden or explained away.

**2. Energy estimation precision**

If US estimates of PRC energy consumption have wide error bars, covert datacenter energy usage might be indistinguishable from estimation noise. Precise energy tracking makes large covert operations more detectable.

## Calibration

**Historical context**: Intelligence estimates of weapons counts from 1960–2025 show median errors of approximately 15%.

**Energy estimation methods**:
- Satellite imagery of power plants
- Trade data for fuel imports
- Economic modeling
- Grid infrastructure analysis

We estimate US intelligence has approximately **7%** median error in estimating PRC datacenter energy consumption.

## Selection Reasoning

The 7% estimate reflects that:
- Energy consumption can be observed through multiple independent methods
- However, China's grid is complex with many industrial facilities
- Attribution of energy to specific facilities (especially hidden ones) is challenging
- Covert facilities might use off-grid power sources (local generation, solar) to reduce signatures
