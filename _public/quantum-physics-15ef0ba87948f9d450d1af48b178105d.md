---
title: "Quantum physics for algorithm implementations"
description: "States, measurements, dynamics, channels, and the physical meaning of an algorithm's output."
---

Quantum algorithms use quantum physics to transform information and extract a specified result. This appendix introduces the finite-dimensional language used throughout the handbook. Read it alongside the [mathematics primer](mathematics.md), then follow the quantities you actually need to reproduce an algorithm.

## States and observables

A pure state is a normalized complex vector $|\psi\rangle$. For one qubit,

$$
|\psi\rangle=\alpha|0\rangle+\beta|1\rangle,
\qquad |\alpha|^2+|\beta|^2=1.
$$

Relative phases affect interference. Multiplying the whole vector by the same phase leaves its physical state unchanged. A state describes the system; an **observable** describes a measurable quantity. Observables are Hermitian operators, such as the Pauli $Z$ operator or the Hamiltonian $H$, which represents energy.

A density matrix also describes mixtures and subsystems entangled with an environment:

$$
\rho=\sum_j p_j|\psi_j\rangle\langle\psi_j|,
\qquad p_j\geq0,\quad \sum_jp_j=1.
$$

It is positive semidefinite and has trace one. Pure states have $\rho=|\psi\rangle\langle\psi|$; different ensembles can describe the same mixed state. Ignoring part of a joint system gives its reduced density matrix by a partial trace. [IBM's density-matrix lesson](https://quantum.cloud.ibm.com/learning/en/courses/general-formulation-of-quantum-information/density-matrices/density-matrix-basics) develops this broader description.

For example, $|+\rangle=(|0\rangle+|1\rangle)/\sqrt2$ and the mixture $\rho=I/2$ both give equal computational-basis probabilities. They differ when measured in the $X$ basis: the pure state always gives the $+1$ outcome, whereas the mixture gives either outcome equally often. Matching one measurement distribution therefore does not establish that two states agree.

## What a measurement returns

For an observable $A=\sum_a aP_a$, a projective measurement returns eigenvalue $a$ with **Born-rule** probability

$$
p(a)=\operatorname{Tr}(P_a\rho),
\qquad \langle A\rangle=\operatorname{Tr}(A\rho).
$$

The projectors $P_a$ include any degeneracy. Measuring a pure state in the computational basis gives $p(x)=|\langle x|\psi\rangle|^2$. One shot produces an outcome; estimating an expectation generally requires repeated preparations and measurements.

For the physical connection between eigenvectors and measurement, watch Barton Zwiebach's [MIT 8.04 video, “Completeness of eigenvectors and measurement postulate”](https://ocw.mit.edu/courses/8-04-quantum-physics-i-spring-2016/resources/completeness-of-eigenvectors-and-measurement-postulate/). Its official lesson page includes the video and transcript.

## Dynamics and open systems

Closed-system evolution is unitary:

$$
\rho(t)=U(t)\rho(0)U(t)^\dagger,\qquad
U(t)=e^{-iHt/\hbar}
$$

for a time-independent Hamiltonian. Circuit gates approximate or compose such transformations; the simulation chapters usually set $\hbar=1$.

Noise, discarded ancillas, and measurements averaged over their outcomes require quantum channels. A completely positive, trace-preserving map has a Kraus representation

$$
\mathcal E(\rho)=\sum_jK_j\rho K_j^\dagger,
\qquad \sum_jK_j^\dagger K_j=I.
$$

Complete positivity ensures valid states even when the input is entangled with another system. Trace preservation retains total probability. A selected measurement branch can instead decrease the trace; that trace records the branch's success probability. [IBM's channel representations](https://quantum.cloud.ibm.com/learning/en/courses/general-formulation-of-quantum-information/quantum-channels/representations-of-channels) explain the channel viewpoint and its implementations.

## Thermal targets and accuracy

For a finite-dimensional Hamiltonian, the Gibbs state at inverse temperature $\beta=1/(k_BT)$ is

$$
\rho_\beta=\frac{e^{-\beta H}}{Z(\beta)},
\qquad Z(\beta)=\operatorname{Tr}(e^{-\beta H}).
$$

The partition function normalizes the state; exponentiating a Hamiltonian alone does not produce a normalized density matrix. Energy eigenstates receive weights proportional to $e^{-\beta E_j}$. See David Tong's [statistical-physics notes on the canonical ensemble](https://www.damtp.cam.ac.uk/user/tong/statphys/statmechhtml/S1.html).

Choose an output metric before estimating resources. A search task needs a correct answer with stated probability. An energy task needs observable accuracy and sampling confidence. A state-preparation task may use trace distance $D(\rho,\sigma)=\tfrac12\|\rho-\sigma\|_1$, which bounds changes in measurement-event probabilities. Fidelity measures similarity; specify whether it is squared or unsquared, as [IBM's fidelity lesson](https://quantum.cloud.ibm.com/learning/en/courses/general-formulation-of-quantum-information/purifications-and-fidelity/fidelity) emphasizes. The [resource baseline](../algorithms/grover.md#end-to-end-resource-analysis) accounts separately for algorithmic approximation, sampling requirements, and physical faults.
