OPENQASM 3.0;
include "stdgates.inc";
gate mcphase(_gate_p_0) _gate_q_0, _gate_q_1 {
  cp(-pi/2) _gate_q_0, _gate_q_1;
}
gate mcphase_0(_gate_p_0) _gate_q_0, _gate_q_1 {
  cp(-pi) _gate_q_0, _gate_q_1;
}
gate mcphase_1(_gate_p_0) _gate_q_0, _gate_q_1 {
  cp(-3*pi/2) _gate_q_0, _gate_q_1;
}
bit[3] c;
qubit[3] q;
reset q[0];
reset q[1];
reset q[2];
ry(0.44279205610415484) q[2];
x q[2];
cry(1.0684520891358287) q[2], q[1];
x q[2];
cry(0.6582813850044836) q[2], q[1];
x q[2];
mcphase(-pi/2) q[1], q[2];
x q[2];
x q[1];
mcphase_0(-pi) q[1], q[2];
x q[1];
mcphase_1(-3*pi/2) q[1], q[2];
x q[2];
ccx q[1], q[2], q[0];
x q[2];
x q[1];
ccx q[1], q[2], q[0];
x q[1];
ccx q[1], q[2], q[0];
x q[1];
ccx q[1], q[2], q[0];
x q[1];
ccx q[1], q[2], q[0];
ccx q[1], q[2], q[0];
cry(-0.6582813850044836) q[2], q[1];
x q[2];
cry(-1.0684520891358287) q[2], q[1];
x q[2];
ry(-0.44279205610415484) q[2];
c[0] = measure q[1];
c[1] = measure q[2];
c[2] = measure q[0];
