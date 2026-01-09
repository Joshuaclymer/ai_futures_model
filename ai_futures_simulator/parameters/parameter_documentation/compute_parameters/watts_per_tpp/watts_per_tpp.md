# watts_per_tpp

| Modal value | Description |
|-------------|-------------|
| 0.0016 W/TPP | Power consumption per unit of transistor performance product for modern chips |

**How we chose this parameter**

This parameter captures the relationship between chip power consumption and compute performance. Energy efficiency improves with smaller process nodes, but the relationship changed significantly after Dennard scaling ended around 2006.

Before 2006, when Dennard scaling was in effect, watts per TPP scaled as (density)^-0.5, meaning that doubling transistor density roughly halved power consumption per unit of performance.

After 2006, watts per TPP scales as (density)^-0.15, meaning modern chips are more power-hungry per transistor than earlier generations would have predicted. This is because voltage can no longer decrease proportionally with transistor size.

The current value of 0.0016 W/TPP represents typical power efficiency for state-of-the-art AI accelerators like the H100.
