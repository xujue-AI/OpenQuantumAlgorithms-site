# Fixed two-round finite composition

This public projection records one accepted fixed finite composition for
`H=0.6Z+0.8X` in Z-then-X order, proposal Z, beta=1, M=4, r=1,
delta=tau=1/8, two rounds and known-plus input. It is an original circuit and
finite-reference example, with no Gibbs-preparation or mixing guarantee.

The accepted check used two weak-dilation Statevector columns and four
external density stages: weak reset prefix and reset-free correction LCU,
first on the initial state and then on the normalized accepted intermediate
state. The full conditional trial is supplied as literal source; it was not
simulated as a measurement-and-branch program. No shots or retry trajectories
were drawn. Sixteen internal reset-related dispatches are predicted from the
pinned backend, and were not measured.

All 66 saved finite checks passed. `summary.json` selects the already-recorded
scalars and scope; `results.json` retains the complete original scientific
`check`, including all eight weak Kraus blocks, all 32 saved correction blocks,
raw density stages, terminal factors, matrices, residuals and tolerances.
Correction failure blocks are authenticated retained gate-simulation data,
not fresh independent failure simulations. Actual p1 and conditional p2 use
their own block traces; p1*p2 is retained separately from the ideal p_C^2.

The Z discrepancy against the finite Psi^2 target is about 6.90e-8, within the
ideal correction-only allowance 4b about 3.75e-6. This allowance excludes
weak-step, splitting, mesh and Trotter errors. The saved Gibbs Frobenius
distance about 0.9273 and generator Gibbs residual about 0.0703 provide no
Gibbs or mixing guarantee. Floating agreement is not a rounding, synthesis
or hardware certificate. N and R remain unevaluated formulas; no single-step
repetition values are reused and no sampling guarantee for a physical device
is claimed.

The raw `whole_invocation_validated`, `all_components_verified` and
`new_algorithm_result` flags remain false. General sampler/channel/correction
evolution, exact finite or continuum KMS, mixing, rotation synthesis,
physical resources, formal proof and floating-point/hardware certification
flags also remain false. Parent acceptance of this finite example is stated
separately from those original flags.

## Files and provenance

- `results.json`: a curated scientific projection; numerical arrays and
  scientific flags are preserved without recomputation, clipping or phase
  fitting. Operational environment and execution metadata are excluded.
- `summary.json`: metadata and selected saved scalars, copied without new
  numerical work. Its provenance names the original accepted source identity
  and original result hash/size; the original result contained additional
  operational metadata and is not this projected JSON.
- `sources/`: seven byte-identical original construction/reference modules.
  Their historical source-only/unexecuted docstrings and source-accounting
  flags are retained. The later accepted staged check is described in the
  data, rather than by rewriting those sources. These are component/reference
  sources, not a standalone experiment runner. The numerical producer and
  operational controller are outside this projection.
- `circuits/`: six byte-identical accepted QASM exports. The conditional
  `two_round_trial.qasm` contains the source branch structure; `round_macro`
  is not a unitary controlled on a quantum success flag.
- `two-round-circuit.svg`: an original schematic view of the literal source,
  with the two herald decisions and terminal reset/readout paths visible.
- `manifest.json` and `manifest.sha256`: identities for the closed selected
  public inventory. The manifest and sidecar are separate from its file list.
- `requirements-lock.txt`, `LICENSE.txt` and `UPSTREAM-LICENSES.md`: the exact
  locked environment and original MIT/upstream package credits.

Parent acceptance authenticated 23 original payloads totaling 1,755,875 bytes.
That original inventory differs from this curated public inventory. The public
manifest binds the projected files separately. Old measured baselines and
original evidence remain unchanged.

Source accounting reserves six quantum wires and eleven classical bits,
with 6,051 static / 6,049 maximum-path / 3,026 early-abort primitive instances
and two classical decisions. Every whole attempt includes preparation,
inverse access, heralds and terminal reset/readout; repetition formulas charge
N*R*6,049 maximum-path primitives. These source counts do not assign physical
qubits or runtime. Classical preprocessing/bookkeeping, arbitrary-angle
synthesis, idle/hold/controller/reset/readout timing, code/factory/routing and
physical faults remain unpriced. No scalable generator or hardware advantage
is established by this one fixed input.

The source specification uses Chen, Kastoryano, Brandao and Gilyen,
*Efficient quantum thermal simulation*, Nature 646 (2025), Eq. 19,
DOI 10.1038/s41586-025-09583-x. Explanations and circuit sources are original;
no upstream benchmark has been reproduced. Locked package versions and
licenses are credited in the adjacent files.

The proposed stdlib display at `scripts/display_thermal_two_round_saved.py`
reads only this manifest and summary, authenticates their pinned metadata,
and prints already-saved values. It performs no numerical replay and does
not verify every bundle payload. Its execution and the associated static
tests remain pending review; this projection has not been built or published.
