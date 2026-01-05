# Covert Unconcealed Capacity

## Purpose

Represents the datacenter capacity that can be diverted to a covert project from facilities that were not originally built for concealment.

## Parameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| Fraction of datacenter capacity to divert | 50% | 0–100% | Proportion of existing (unconcealed) datacenter capacity redirected to covert operations |

## How It Works

When a covert project begins, it may not have time to construct purpose-built concealed facilities. Instead, it can repurpose existing datacenter infrastructure:

- **Existing commercial datacenters**: Can be commandeered or have portions allocated to government use
- **Government facilities**: May have spare capacity that can be redirected
- **Research institutions**: University and national lab computing resources

These facilities were not designed for concealment, making them more vulnerable to satellite and energy-based detection methods.

## Trade-offs

**Benefits of diverting existing capacity**:
- Immediate availability (no construction time)
- Proven infrastructure and power supply
- Existing workforce and operational expertise

**Risks of diverting existing capacity**:
- Higher detection probability (visible to satellites)
- Energy consumption patterns may be noticed
- Workforce not vetted for covert operations

## Selection Reasoning

The 50% default reflects a moderate scenario where:
- The covert project needs substantial compute quickly
- But complete commandeering of all facilities would be too conspicuous
- A balance is struck between capability and concealment
