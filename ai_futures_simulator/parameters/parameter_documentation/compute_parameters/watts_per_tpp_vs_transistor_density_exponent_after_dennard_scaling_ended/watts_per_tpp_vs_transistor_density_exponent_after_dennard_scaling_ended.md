# watts_per_tpp_vs_transistor_density_exponent_after_dennard_scaling_ended

| Modal value | Description |
|-------------|-------------|
| -0.15 | Exponent relating watts per TPP to transistor density after Dennard scaling ended |

**How we chose this parameter**

Energy efficiency scales with transistor density according to a power law: W/TPP ∝ (transistor density)^exponent. After Dennard scaling ended around 2006, voltage could no longer decrease with transistor size because it had reached physical limits. The exponent became less negative, reducing efficiency gains from further miniaturization.

![Energy efficiency vs transistor density](/parameter_docs/compute_parameters/transistor_density_at_end_of_dennard_scaling_m_per_mm2/energy_efficiency_vs_transistors.png)

With an exponent of -0.15, doubling transistor density now only improves energy efficiency by about 10%, compared to the 30% improvement that was possible before 2006. This change fundamentally altered the economics of chip scaling and is why power consumption has become an increasingly binding constraint on AI compute.

Source: [Epoch AI Hardware Trends](https://epoch.ai/blog/trends-in-machine-learning-hardware)
