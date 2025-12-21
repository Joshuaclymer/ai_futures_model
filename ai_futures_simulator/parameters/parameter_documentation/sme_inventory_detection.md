# SME Inventory Detection

## Purpose

Models the probability that US intelligence detects a covert fab by identifying discrepancies between China's declared semiconductor manufacturing equipment (SME) inventory and US estimates.

## Parameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| Intelligence median error (fab stock) | 7% | 1–50% | Median percentage error in US estimates of PRC semiconductor manufacturing equipment |

## How Detection Works

Detection probability depends on two factors:

**1. Diversion fraction**

Large diversions are conspicuous. Diverting 80 of 100 lithography scanners would be obvious; diverting a small fraction might remain within estimation error.

**2. Estimation precision**

If US estimates have wide error bars (e.g., "50–150 DUV scanners"), small discrepancies are indistinguishable from noise. Precise estimates make even small diversions detectable.

## Calibration

**Historical context**: Intelligence estimates of weapons counts from 1960–2025 show median errors of approximately 15%.

**Equipment vs. weapons**: China's semiconductor industry has characteristics that improve visibility:
- Quasi-private companies with some investor reporting
- Extensive equipment supply chain with few suppliers (ASML, Applied Materials, etc.)
- Export controls create tracking infrastructure
- Equipment requires maintenance and parts from original suppliers

We estimate US intelligence error at approximately **7%**—roughly 2x better than for weapons programs.

## Selection Reasoning

The 7% estimate reflects that:
- Lithography equipment is highly specialized and traceable
- Few global suppliers (near-monopoly for EUV)
- However, China has stockpiled equipment and may have unreported inventory
- Domestic equipment production adds uncertainty
