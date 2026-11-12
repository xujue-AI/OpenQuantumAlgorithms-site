---
title: "Quantum simulation"
description: "Explicit dynamics circuits, with open-system and Gibbs preparation tracks planned."
---

Quantum simulation asks for a specified state transformation or an observable of a physical model. This umbrella separates three tasks because their inputs, quantum circuits, accuracy requirements and readout costs differ.

| Track | Task | Current evidence |
|---|---|---|
| [Hamiltonian dynamics](hamiltonian-simulation.md) | Approximate $e^{-iHt}$ or measure a stated observable after evolution | Checked first/second product formulas; explicit postselected Taylor LCU; checked QSP cosine primitive |
| [Open systems](simulation/open-systems.md) | Approximate a channel or Lindblad evolution on a density operator | Planned; no channel/dilation circuit or validated result yet |
| [Thermal/Gibbs preparation](simulation/gibbs-preparation.md) | Prepare $e^{-\beta H}/\operatorname{Tr}(e^{-\beta H})$ or estimate a thermal observable | Planned; no preparation circuit or validated result yet |

The first dynamics example fixes the open spin chain, its initial state, time and observable. Its small dense classical answer checks the implemented circuit. Parameterized gate generators have separate materialization limits; a tiny successful simulation does not establish a computational advantage.

The QSP example currently implements a cosine filter, and the Taylor LCU example uses explicit postselection. Neither supplies a completed amplified, fault-tolerant dynamics pipeline. Their individual status and missing costs are stated on the dynamics page.
