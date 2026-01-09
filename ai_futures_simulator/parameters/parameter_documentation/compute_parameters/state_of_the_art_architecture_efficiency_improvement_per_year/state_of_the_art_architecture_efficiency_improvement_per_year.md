# state_of_the_art_architecture_efficiency_improvement_per_year

| Modal value | Description |
|-------------|-------------|
| 1.23x per year | Annual improvement in chip performance from architecture optimization (excluding transistor scaling) |

**How we chose this parameter**

Chip performance improves through two mechanisms: increased transistor density and improved chip architecture. This parameter captures architecture improvements—performance gains from design optimization rather than transistor scaling.

According to Epoch's data, total chip performance (TPP per die area) has improved at approximately 1.55x per year:

![TPP per die area over time](/parameter_docs/compute_parameters/state_of_the_art_architecture_efficiency_improvement_per_year/tpp_per_die_area.png)

[Download data](/api/parameter_docs/compute_parameters/state_of_the_art_architecture_efficiency_improvement_per_year/epoch_data.csv)

Transistor density has increased at approximately 1.26x per year:

![Transistor density over time](/parameter_docs/compute_parameters/state_of_the_art_architecture_efficiency_improvement_per_year/transistor_density_over_time.png)

The ratio (1.55 / 1.26 = 1.23x per year) represents architecture improvements alone. This decomposition allows the model to separately track gains from manufacturing process improvements and gains from design innovation.
