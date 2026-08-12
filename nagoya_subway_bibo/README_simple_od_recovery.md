# Simple OD–recovery experiment

`run_simple_od_recovery.py` must be placed in the same directory as
`fare_model.py`, `od_model.py`, and `bibo_link_inference_model.py`.

The experiment deliberately uses no hub boost and RFID only. Defaults:

- gamma: `0.18` (short-trip), `0.0` (neutral), `-0.10` (long-trip)
- RFID link detection rate: `0.65`, `0.70`
- 30 seeds, 3,300 passengers per seed
- confidence threshold: `0.30`

```bash
python3 run_simple_od_recovery.py
```

Outputs are written to `simple_od_recovery_outputs/`:

- `overall_recovery.csv`
- `recovery_by_trip_length.csv`
- `overall_recovery_comparison.png`
- `recovery_by_trip_length_comparison.png`
- `run_config.json`

The two route quantities are intentionally separate:

```text
d_ij = sum of commercial link distances on route Gamma_ij
m_ij = number of links in route Gamma_ij
```

OD demand is `exp(-gamma * d_ij)`, while RFID observation opportunities equal
`m_ij`.
