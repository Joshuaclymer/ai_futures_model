# AI Chip Lifespan

This parameter controls the rate at which AI chips fail over time. Empirical data on GPU failure rates is limited, but three sources inform our estimate:

## Meta's Llama 3 Training

Meta trained Llama 3 on 16,000 H100s over 21 million GPU-hours. Of 419 interruptions, only 3 required physical intervention—most failures were software-related.

Source: [Meta's Llama 3 Paper](https://arxiv.org/pdf/2407.21783)

## Blue Waters (2013–2021)

This supercomputer operated 4,000 K20X GPUs under heavy use for 8 years. Approximately 97% survived the full period.

Source: [Blue Waters Study](https://cug.org/proceedings/cug2021_proceedings/includes/files/pap102s2-file1.pdf)

## Titan (2012–2018)

Cooling significantly affects longevity. Among well-cooled GPUs, 60–90% survived 5 years; poorly-cooled units showed 10–40% survival rates.

Source: [Titan GPU Study](https://christian-engelmann.de/publications/ostrouchov20gpu.pdf)
