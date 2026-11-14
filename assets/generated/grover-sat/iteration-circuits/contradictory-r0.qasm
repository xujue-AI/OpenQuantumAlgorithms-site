OPENQASM 3.0;
include "stdgates.inc";
bit[2] assignment;
qubit[2] x_0;
qubit[2] clause;
h x_0[0];
h x_0[1];
assignment[0] = measure x_0[0];
assignment[1] = measure x_0[1];
