# Evaluation Report

- Generated at: `2026-04-24T16:22:46.486857+00:00`
- Dataset: `data/eval/golden_dataset.jsonl`
- Modes: `standard, advanced`
- Dataset size: `8`
- Mode outputs: `16`

## Standard vs Advanced

- Paired examples: `8`
- Avg latency delta (advanced - standard, ms): `4.25`
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
| standard | ambiguous | 1 | 3.0 | 0.5 | 1.000 | 0.000 | 0.000 |
| standard | conflicting_sources | 1 | 3.0 | 0.5 | 1.000 | 0.000 | 0.000 |
| standard | insufficient_context | 1 | 3.0 | 0.5 | 1.000 | 0.000 | 0.000 |
| standard | multi_hop | 2 | 3.0 | 0.5 | 1.000 | 0.000 | 0.000 |
| standard | simple | 1 | 3.0 | 0.5 | 1.000 | 0.000 | 0.000 |
| standard | vietnamese | 2 | 3.0 | 0.5 | 1.000 | 0.000 | 0.000 |
| advanced | ambiguous | 1 | 7.0 | 0.2 | 1.000 | 0.000 | 1.000 |
| advanced | conflicting_sources | 1 | 9.0 | 0.2 | 1.000 | 0.000 | 1.000 |
| advanced | insufficient_context | 1 | 7.0 | 0.2 | 1.000 | 0.000 | 1.000 |
| advanced | multi_hop | 2 | 7.0 | 0.51 | 1.000 | 0.000 | 1.000 |
| advanced | simple | 1 | 7.0 | 0.2 | 1.000 | 0.000 | 1.000 |
| advanced | vietnamese | 2 | 7.0 | 0.51 | 1.000 | 0.000 | 1.000 |

## Notes

- `groundedness_proxy` is a lexical-overlap proxy only and not a perfect groundedness metric.