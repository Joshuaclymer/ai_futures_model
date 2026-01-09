# energy_efficiency_of_compute_stock_relative_to_state_of_the_art

| Modal value | Description |
|-------------|-------------|
| 0.5 | Energy efficiency of covert project chips relative to state-of-the-art (H100) |

**How we chose this parameter**

This multiplier represents the energy efficiency of the chips available to a covert project relative to an H100. A value of 0.5 means the covert project's chips consume twice the energy per unit of compute compared to an H100. A value of 1.0 would indicate equivalent efficiency, and 2.0 would mean twice as efficient.

We use 0.5 as the default because a covert project would likely rely on domestically-produced Chinese chips or older-generation imported hardware, both of which are less energy-efficient than NVIDIA's latest designs. This parameter directly affects the power requirements and thermal signatures of covert datacenters, which in turn affects detectability.
