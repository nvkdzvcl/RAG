# Evaluation Report

- Generated at: `2026-04-24T11:29:49.326057+00:00`
- Dataset: `data/eval/golden_dataset.jsonl`
- Modes: `standard, advanced, compare`
- Dataset size: `8`
- Mode outputs: `24`

## Standard vs Advanced

- Paired examples: `8`
- Avg latency delta (advanced - standard, ms): `8.125`
- Avg confidence delta (advanced - standard): `-0.14500000000000002`
- Advanced retry rate: `1.000`

## Rates

- Abstain rate (standard): `0.000`
- Abstain rate (advanced): `0.000`
- Citation rate (standard): `1.000`
- Citation rate (advanced): `1.000`

## Per-Category Summary

| mode | category | count | avg_latency_ms | avg_confidence | citation_rate | abstain_rate | retry_rate |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| standard | ambiguous | 1 | 5.0 | 0.5 | 1.000 | 0.000 | 0.000 |
| standard | conflicting_sources | 1 | 12.0 | 0.5 | 1.000 | 0.000 | 0.000 |
| standard | insufficient_context | 1 | 11.0 | 0.5 | 1.000 | 0.000 | 0.000 |
| standard | multi_hop | 2 | 4.0 | 0.5 | 1.000 | 0.000 | 0.000 |
| standard | simple | 1 | 4.0 | 0.5 | 1.000 | 0.000 | 0.000 |
| standard | vietnamese | 2 | 10.0 | 0.5 | 1.000 | 0.000 | 0.000 |
| advanced | ambiguous | 1 | 9.0 | 0.2 | 1.000 | 0.000 | 1.000 |
| advanced | conflicting_sources | 1 | 19.0 | 0.2 | 1.000 | 0.000 | 1.000 |
| advanced | insufficient_context | 1 | 18.0 | 0.2 | 1.000 | 0.000 | 1.000 |
| advanced | multi_hop | 2 | 21.0 | 0.51 | 1.000 | 0.000 | 1.000 |
| advanced | simple | 1 | 18.0 | 0.2 | 1.000 | 0.000 | 1.000 |
| advanced | vietnamese | 2 | 9.5 | 0.51 | 1.000 | 0.000 | 1.000 |

## Notes

- `groundedness_proxy` is a lexical-overlap proxy only and not a perfect groundedness metric.