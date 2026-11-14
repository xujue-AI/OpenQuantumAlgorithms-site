# Reproduce bounded linear-system sweeps

Use Python 3.12 and install the supplied requirements lock. The generator imports the
included `openquantum` package directly; an editable install is unnecessary.

```bash
python generate_linear_systems_physical.py --no-estimate
python generate_linear_systems_physical.py
```

The second command makes at most 13 local QDK queries, using an external private cache.
Each case is a separate normalized-solution Z task. These are conditional mean-fit
resource models, with no synthesized or hardware output-confidence certificate.
Origin default-artifact hashes are retained, while only baseline source copies present
in this download can be rechecked. Original code is MIT; exact references are included.
