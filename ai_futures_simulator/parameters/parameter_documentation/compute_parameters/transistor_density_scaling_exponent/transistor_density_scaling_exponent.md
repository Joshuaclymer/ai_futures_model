# transistor_density_scaling_exponent

| Modal value | Description |
|-------------|-------------|
| 1.49 | Exponent describing how transistor density scales with process node |

**How we chose this parameter**

Transistor density depends on process node—the resolution at which circuits are etched onto silicon. Smaller nodes increase transistor count per chip, yielding more compute per wafer. The relationship between process node and transistor density follows a power law, and this parameter captures the exponent of that relationship.

![H100 equivalents vs process node](/parameter_docs/compute_parameters/transistor_density_scaling_exponent/h100_equiv_per_year_vs_process_node.png)

[Download data](/api/parameter_docs/compute_parameters/transistor_density_scaling_exponent/epoch_data.csv)

The chart shows H100-equivalents producible per wafer at different process nodes (assuming 28 H100-sized chips per wafer). Moving from 28nm to 7nm approximately quadruples output. The exponent of 1.49 was fitted to historical data from Epoch AI on chip performance across different process nodes.
