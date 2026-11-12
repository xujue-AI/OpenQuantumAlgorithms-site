---
title: "Mathematics for quantum algorithms"
description: "The linear algebra, approximation, and probability needed to read circuits and resource estimates."
---

The mathematical task is to connect a problem specification to a circuit and a checkable output. This primer collects notation used by the algorithm chapters. It complements the [quantum-physics primer](quantum-physics.md); no background in formal proof software is required.

## Vectors, inner products, and registers

A ket $|v\rangle$ is a complex column vector; its bra $\langle v|$ is the conjugate transpose. The inner product and Euclidean norm are

$$
\langle v|w\rangle=\sum_j\overline{v_j}w_j,
\qquad \|v\|_2=\sqrt{\langle v|v\rangle}.
$$

Orthogonal vectors have zero inner product. An orthonormal basis gives coordinates without changing lengths. Normalized state vectors have unit norm, while input data vectors may require a separate normalization factor. [IBM's inner-products lesson](https://quantum.cloud.ibm.com/learning/en/courses/basics-of-quantum-information/quantum-circuits/inner-products-and-projections) develops these conventions.

Two registers combine by a tensor product: $|v\rangle\otimes|w\rangle$. Dimensions multiply, so $n$ qubits have $2^n$ basis states. A joint vector need not factor into individual-register vectors; this is the mathematical possibility behind entanglement. Register order and bit order must match the generator, simulator, and classical comparison. A statevector containing every amplitude therefore has very different storage requirements from a circuit description.

## Operators and spectra

Matrices act linearly on vectors. The adjoint $A^\dagger$ conjugates and transposes; a unitary satisfies $U^\dagger U=I$ and preserves inner products. A Hermitian matrix satisfies $H^\dagger=H$ and has real eigenvalues with an orthonormal eigenbasis:

$$
H|u_j\rangle=\lambda_j|u_j\rangle,\qquad
H=\sum_j\lambda_j|u_j\rangle\langle u_j|.
$$

[Phase estimation](../algorithms/qft-qpe.md) targets the phase in $U|u_j\rangle=e^{2\pi i\phi_j}|u_j\rangle$. The prepared state's overlap with the relevant eigenvector affects the chance of obtaining that phase.

Watch Gilbert Strang's [MIT 18.06 video, “Lecture 21: Eigenvalues and eigenvectors”](https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/resources/lecture-21-eigenvalues-and-eigenvectors/) for the underlying eigenvector equation. The official lesson includes a video and transcript; quantum phase estimation applies this idea to unitary operators.

## Norms and approximations

A norm makes “close” quantitative. The operator norm

$$
\|A\|=\sup_{\|v\|_2=1}\|Av\|_2
$$

measures the largest amplification of a vector. The trace norm $\|A\|_1$ is the sum of singular values, and the Frobenius norm $\|A\|_F=\sqrt{\operatorname{Tr}(A^\dagger A)}$ measures all entries together. These norms give different error guarantees.

For Hermitian $H$, define $f(H)$ by applying $f$ to its eigenvalues. If a polynomial $p$ approximates $f$ uniformly on an interval containing the spectrum, then

$$
\sup_{x\in[a,b]}|p(x)-f(x)|\leq\epsilon
\quad\Longrightarrow\quad
\|p(H)-f(H)\|\leq\epsilon.
$$

This motivates polynomial methods for evolution and inversion. QSP/QSVT circuits also impose encoding, boundedness, and parity conditions; an arbitrary polynomial fit alone is insufficient. [PennyLane's QSVT introduction](https://pennylane.ai/demos/tutorial_intro_qsvt) explains this connection.

[Linear-system algorithms](../algorithms/linear-systems.md) must additionally track conditioning. For an invertible matrix, the spectral condition number is $\kappa=\sigma_{\max}/\sigma_{\min}$. Small singular values can make inversion sensitive and expensive. Producing a state proportional to $A^{-1}b$ also leaves its normalization and requested readout to be accounted for. It does not directly return a classical list of every solution component.

## Probability and confidence

For independent preparations with success probability $p$, $R$ trials all fail with probability $(1-p)^R$. Repetition costs belong in the algorithm estimate.

For independent Bernoulli outcomes $X_1,\ldots,X_M$ with common mean $p$, the frequency $\hat p=M^{-1}\sum_iX_i$ estimates that probability. Hoeffding's bound gives

$$
\Pr(|\hat p-p|\geq\epsilon)\leq2e^{-2M\epsilon^2}.
$$

Thus $M\geq\ln(2/\delta)/(2\epsilon^2)$ suffices for confidence at least $1-\delta$. [John Duchi's Stanford notes](https://cs229.stanford.edu/extra-notes/hoeffding.pdf) state the independence and boundedness assumptions. Sampling confidence controls random variation; it does not remove a biased circuit or an inaccurate model. Specify those errors separately when comparing quantum and classical outputs.
