# transistor_density_at_end_of_dennard_scaling_m_per_mm2

| Modal value | Description |
|-------------|-------------|
| 1.98 million/mm² | Transistor density at which Dennard scaling ended (approximately 2006) |

**How we chose this parameter**

Dennard scaling—the principle that smaller transistors use proportionally less power—enabled decades of efficiency improvements. As transistors shrank, operating voltages could decrease, allowing more transistors without increased power consumption.

Around 2006, this relationship broke down as voltages approached physical limits. The chart below shows this inflection point, where the relationship between transistor density and energy efficiency changed slope:

![Energy efficiency vs transistor density](/parameter_docs/compute_parameters/transistor_density_at_end_of_dennard_scaling_m_per_mm2/energy_efficiency_vs_transistors.png)

[Download data](/api/parameter_docs/compute_parameters/transistor_density_at_end_of_dennard_scaling_m_per_mm2/epoch_data.csv)

This parameter specifies the transistor density (millions per mm²) at which Dennard scaling ended. Before this point, energy efficiency improved rapidly with transistor density. After this point, efficiency gains have been more modest, which affects projections of future chip performance and power consumption.

Source: [Epoch AI Hardware Trends](https://epoch.ai/blog/trends-in-machine-learning-hardware)
