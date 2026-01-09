# initial_annual_hazard_rate

| Modal value | Description |
|-------------|-------------|
| 3% per year | Annual probability that an AI chip fails and must be replaced |

**How we chose this parameter**

This parameter controls the rate at which AI chips fail over time. Empirical data on GPU failure rates is limited, but three sources inform our estimate.

Meta trained Llama 3 on 16,000 H100s over 21 million GPU-hours. Of 419 interruptions, only 3 required physical intervention—most failures were software-related. This suggests very low hardware failure rates under production conditions. (Source: [Meta's Llama 3 Paper](https://arxiv.org/pdf/2407.21783))

The Blue Waters supercomputer operated 4,000 K20X GPUs under heavy use for 8 years from 2013 to 2021. Approximately 97% survived the full period, implying an annual hazard rate of less than 1%. (Source: [Blue Waters Study](https://cug.org/proceedings/cug2021_proceedings/includes/files/pap102s2-file1.pdf))

The Titan supercomputer from 2012 to 2018 showed that cooling significantly affects longevity. Among well-cooled GPUs, 60–90% survived 5 years; poorly-cooled units showed 10–40% survival rates. (Source: [Titan GPU Study](https://christian-engelmann.de/publications/ostrouchov20gpu.pdf))

Based on this evidence, we estimate a 3% annual hazard rate, which implies most chips will last 10+ years under good operating conditions. This is conservative compared to the Blue Waters data but accounts for the fact that covert facilities might have less optimal cooling and maintenance.
