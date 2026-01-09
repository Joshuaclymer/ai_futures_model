# h100_power_watts

| Modal value | Description |
|-------------|-------------|
| 700 watts | Power consumption of a single NVIDIA H100 GPU during operation |

**How we chose this parameter**

An NVIDIA H100 GPU consumes approximately 700 watts during operation (TDP - Thermal Design Power). This serves as the baseline for datacenter power requirement calculations. Combined with the energy efficiency parameter, this allows conversion between H100-equivalent compute units and actual power consumption.

Source: [NVIDIA H100 Specifications](https://www.nvidia.com/en-us/data-center/h100/)
