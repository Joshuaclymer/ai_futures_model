# Chip Stock Detection

## Purpose

Models the probability that US intelligence detects a covert project by identifying discrepancies between China's reported AI chip stock and US estimates.

## Parameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| Intelligence median error (chip stock) | 7% | 1–50% | Median percentage error in US estimates of PRC AI chip inventory |

## How Detection Works

Detection probability depends on two factors:

**1. Diversion fraction**

Larger diversions are easier to detect. Diverting 50% of AI chips would be conspicuous; diverting 5% might remain within estimation error.

**2. Estimation precision**

If US estimates have wide error bars (e.g., "800K–1.2M H100-equivalents"), small discrepancies are indistinguishable from noise. Precise estimates make even small diversions detectable.

## Calibration

**Historical context**: Intelligence estimates of weapons counts from 1960–2025 show median errors of approximately 15%.

**AI chips vs. weapons**: AI chips flow through a highly specialized supply chain that may be easier to monitor than weapons programs:
- Concentrated manufacturing (TSMC, Samsung)
- Export controls create tracking infrastructure
- Commercial applications require some transparency

We estimate US intelligence error at approximately **7%**—roughly half the historical weapons estimate error.

## Selection Reasoning

The 7% estimate reflects that:
- The semiconductor supply chain is more visible than weapons programs
- Export controls have created monitoring infrastructure
- However, China has domestic production and smuggling routes that add uncertainty
