# watts_per_tpp_vs_transistor_density_exponent_before_dennard_scaling_ended

| Modal value | Description |
|-------------|-------------|
| -0.5 | Exponent relating watts per TPP to transistor density before Dennard scaling ended |

**How we chose this parameter**

Energy efficiency scales with transistor density according to a power law: W/TPP ∝ (transistor density)^exponent. Before Dennard scaling ended around 2006, voltage could decrease with transistor size, enabling substantial efficiency gains. The exponent was approximately -0.5, meaning that doubling transistor density roughly halved power consumption per unit of performance.

![Energy efficiency vs transistor density](/parameter_docs/compute_parameters/transistor_density_at_end_of_dennard_scaling_m_per_mm2/energy_efficiency_vs_transistors.png)

This relationship drove decades of rapid efficiency improvements in computing. The change in this exponent after 2006 is why modern chips require increasingly sophisticated cooling solutions despite incremental improvements in transistor density.

Source: [Epoch AI Hardware Trends](https://epoch.ai/blog/trends-in-machine-learning-hardware)
