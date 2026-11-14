# Bounded complete simulation workloads

Five actual measured Clifford+T instruments cover dynamics n=1/2 (Z0)
and thermal n=1 (X), n=2 (X and ZZ). Repeated trial populations and hardware
schedules are calculated, unexecuted. The total task error target is 0.05:
source 0.01, realization 0.01, statistics 0.03. Four failure categories each
receive 0.0025; the two thermal n=2 groups share every task allocation equally.

The source is the explicit retargetable QSP/access algorithm. Every attempt
retains input resets, preparation, access/uncomputation, acceptance readout,
thermal reference discard, basis changes and every system readout. Every
rejected complete attempt and capped factory retry is charged. The scorer
uses exact bounded Kraus/statevector semantics, including correlated bits.

Source interval bounds, guarded actual Ross–Selinger numerical synthesis,
a separate 1e-7 SDK/control/routing reserve, Hoeffding confidence and physical
fault hypotheses remain separate. Actual sequence hashes identify the saved
synthesis (the native backend exposes no seed); SABRE uses seed 7. The reserve
is an explicit norm assumption, not inferred from nominal/routed agreement.
The QASM exporter omits overall phase; exports specify measured trials rather
than phase-preserving controlled oracles. Full in-memory synthesis disables
clean-input lowering. No formal synthesis or hardware confidence certificate
is claimed. Classical preprocessing/report wall time is outside quantum time.

Thermal acceptance uses the exact trace-zero Pauli input and Bell Jensen
bound Tr(exp(-beta H))/d>=1, giving target branch norm>=exp(-a_exact)/4.
The weaker generic exp(-2a_exact)/4 control is also recorded. Neither bound
uses the numerical instance acceptance. Each group has a bounded scalar
score; statistical correlations within a measured bitstring are allowed.

This is a separate serialized instrument scenario, preserving the frozen
Grover/PSSPC baseline. Physical error 1e-3/1e-4 is crossed with the unchanged
default controller 1e-15/decision and a separate conditional 1e-18 scenario.
All other profile assumptions are unchanged. Infeasibility remains explicit;
a controller contribution is a bound within the additive engineering model,
not a lower bound on actual hardware failure. Sequential thermal X/Z results
use a common distance/factory size/supply time/output quality. Peak space is
the maximum and runtime/failure allocations add; no incompatible minima are
combined. Distances are odd 3..49 and factory retries are capped at 16.

Limits: one/two systems; eight total wires; 20,000 source instructions;
500,000 synthesized/check instructions; 256 weighted branches; 1e12 complete
attempts. These bounded examples do not establish scalable hardware advantage.
The physical backend assumes tracked frames, protected Pauli bases and a
factory interface; it does not certify geometry, decoder throughput or
arbitrary adaptive feedback. See each model's assumptions/excluded fields.

Reproduce with the locked Python 3.12 environment:

    PYTHONPATH=src python scripts/generate_simulation_complete_resources.py
    PYTHONPATH=src python -m pytest tests/test_simulation_complete_resource_workloads.py tests/test_simulation_instrument_caps.py

Standalone: save every copied Python source, lock, license and references
beside the generator and notebook. The default standalone output is a new
subdirectory. Use --case for one of the five bounded groups. Partial generation
removes only managed omitted case files. --reuse-prepared explicitly reuses
runner-generated private checkpoints keyed by exact runtime source/package
identities. Cache and public output must be disjoint; the private cache is
outside the checkout. No checkpoint is distributed. manifests record source,
package and QASM hashes; standalone revision is null. Original MIT code/text/
figures; primary algorithm/package references are in references.bib.
