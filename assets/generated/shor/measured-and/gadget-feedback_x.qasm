OPENQASM 3.0;
include "stdgates.inc";
bit[1] c;
qubit[4] q;
reset q[3];
h q[3];
t q[3];
cx q[1], q[3];
tdg q[3];
cx q[0], q[3];
t q[3];
cx q[1], q[3];
tdg q[3];
h q[3];
sdg q[3];
cx q[3], q[2];
h q[3];
c[0] = measure q[3];
if (c[0]) {
  h q[1];
  cx q[0], q[1];
  h q[1];
  x q[3];
}
