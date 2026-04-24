# Evaluation Report

- Generated at: `2026-04-24T16:05:11.973527+00:00`
- Dataset: `data/eval/golden.jsonl`
- Modes: `standard, advanced`
- Dataset size: `8`
- Mode outputs: `16`

## Standard vs Advanced

- Paired examples: `8`
- Avg latency delta (advanced - standard, ms): `12.0`
- Avg confidence delta (advanced - standard): `0.16999999999999993`
- Advanced retry rate: `0.000`

## Rates

- Abstain rate (standard): `0.000`
- Abstain rate (advanced): `0.000`
- Citation rate (standard): `1.000`
- Citation rate (advanced): `1.000`

## Per-Category Summary

| mode | category | count | avg_latency_ms | avg_confidence | citation_rate | abstain_rate | retry_rate |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| standard | ambiguous | 1 | 12.0 | 0.55 | 1.000 | 0.000 | 0.000 |
| standard | conflicting_sources | 1 | 12.0 | 0.55 | 1.000 | 0.000 | 0.000 |
| standard | insufficient_context | 1 | 12.0 | 0.55 | 1.000 | 0.000 | 0.000 |
| standard | multi_hop | 2 | 12.0 | 0.55 | 1.000 | 0.000 | 0.000 |
| standard | simple | 1 | 12.0 | 0.55 | 1.000 | 0.000 | 0.000 |
| standard | vietnamese | 2 | 12.0 | 0.55 | 1.000 | 0.000 | 0.000 |
| advanced | ambiguous | 1 | 24.0 | 0.72 | 1.000 | 0.000 | 0.000 |
| advanced | conflicting_sources | 1 | 24.0 | 0.72 | 1.000 | 0.000 | 0.000 |
| advanced | insufficient_context | 1 | 24.0 | 0.72 | 1.000 | 0.000 | 0.000 |
| advanced | multi_hop | 2 | 24.0 | 0.72 | 1.000 | 0.000 | 0.000 |
| advanced | simple | 1 | 24.0 | 0.72 | 1.000 | 0.000 | 0.000 |
| advanced | vietnamese | 2 | 24.0 | 0.72 | 1.000 | 0.000 | 0.000 |

## Notes

- `groundedness_proxy` is a lexical-overlap proxy only and not a perfect groundedness metric.