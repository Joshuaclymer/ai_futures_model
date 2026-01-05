# Satellite Datacenter Detection

## Purpose

Models the probability that US intelligence detects a covert project by identifying discrepancies between China's declared datacenter capacity and satellite-based estimates.

## Parameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| Intelligence median error (satellite) | 1% | 0.1–10% | Median percentage error in satellite-based estimates of PRC datacenter capacity |

## How Detection Works

Detection probability depends on two factors:

**1. Diversion fraction**

The PRC may divert existing datacenter capacity (not built for concealment) to the covert project. Large diversions are conspicuous; small diversions might remain within estimation error.

**2. Satellite estimation precision**

If satellite-based estimates have wide error bars, small discrepancies are indistinguishable from noise. High-resolution satellite imagery and thermal signatures can improve precision, making even small diversions detectable.

## Calibration

**Historical context**: Intelligence estimates of weapons counts from 1960–2025 show median errors of approximately 15%.

**Datacenters vs. weapons**: Physical datacenter infrastructure is relatively easy to observe from space:
- Large building footprints
- Distinctive cooling systems
- Thermal signatures from 24/7 operation
- Power infrastructure (substations, transmission lines)
- Consistent location patterns (near power, fiber, water)

We estimate satellite-based datacenter capacity estimates have approximately **1%** median error for facilities not specifically built for concealment.

## Selection Reasoning

The 1% estimate reflects that:
- Datacenters have distinctive, hard-to-hide physical signatures
- Modern satellite imagery has very high resolution
- Thermal imaging can detect operating facilities even without visible construction
- However, this only applies to existing/unconcealed facilities—purpose-built covert facilities would require different detection methods
