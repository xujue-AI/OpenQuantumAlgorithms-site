OPENQASM 3.0;
include "stdgates.inc";
gate mcphase(_gate_p_0) _gate_q_0, _gate_q_1 {
  cp(3*pi/5) _gate_q_0, _gate_q_1;
}
gate _cP_2pi_phi_ _gate_q_0, _gate_q_1 {
  mcphase(3*pi/5) _gate_q_0, _gate_q_1;
}
gate _cP_2pi_phi__0 _gate_q_0, _gate_q_1 {
  mcphase(3*pi/5) _gate_q_0, _gate_q_1;
}
gate _cP_2pi_phi__1 _gate_q_0, _gate_q_1 {
  mcphase(3*pi/5) _gate_q_0, _gate_q_1;
}
gate _cP_2pi_phi__2 _gate_q_0, _gate_q_1 {
  mcphase(3*pi/5) _gate_q_0, _gate_q_1;
}
gate _cP_2pi_phi__3 _gate_q_0, _gate_q_1 {
  mcphase(3*pi/5) _gate_q_0, _gate_q_1;
}
gate _cP_2pi_phi__4 _gate_q_0, _gate_q_1 {
  mcphase(3*pi/5) _gate_q_0, _gate_q_1;
}
gate _cP_2pi_phi__5 _gate_q_0, _gate_q_1 {
  mcphase(3*pi/5) _gate_q_0, _gate_q_1;
}
gate _cP_2pi_phi__6 _gate_q_0, _gate_q_1 {
  mcphase(3*pi/5) _gate_q_0, _gate_q_1;
}
gate _cP_2pi_phi__7 _gate_q_0, _gate_q_1 {
  mcphase(3*pi/5) _gate_q_0, _gate_q_1;
}
gate _cP_2pi_phi__8 _gate_q_0, _gate_q_1 {
  mcphase(3*pi/5) _gate_q_0, _gate_q_1;
}
gate _cP_2pi_phi__9 _gate_q_0, _gate_q_1 {
  mcphase(3*pi/5) _gate_q_0, _gate_q_1;
}
gate _cP_2pi_phi__10 _gate_q_0, _gate_q_1 {
  mcphase(3*pi/5) _gate_q_0, _gate_q_1;
}
gate _cP_2pi_phi__11 _gate_q_0, _gate_q_1 {
  mcphase(3*pi/5) _gate_q_0, _gate_q_1;
}
gate _cP_2pi_phi__12 _gate_q_0, _gate_q_1 {
  mcphase(3*pi/5) _gate_q_0, _gate_q_1;
}
gate _cP_2pi_phi__13 _gate_q_0, _gate_q_1 {
  mcphase(3*pi/5) _gate_q_0, _gate_q_1;
}
bit[4] c;
qubit[2] q;
reset q[0];
reset q[1];
x q[1];
reset q[0];
h q[0];
_cP_2pi_phi_ q[0], q[1];
_cP_2pi_phi__0 q[0], q[1];
_cP_2pi_phi__1 q[0], q[1];
_cP_2pi_phi__2 q[0], q[1];
_cP_2pi_phi__3 q[0], q[1];
_cP_2pi_phi__4 q[0], q[1];
_cP_2pi_phi__5 q[0], q[1];
_cP_2pi_phi__6 q[0], q[1];
h q[0];
c[0] = measure q[0];
reset q[0];
h q[0];
_cP_2pi_phi__7 q[0], q[1];
_cP_2pi_phi__8 q[0], q[1];
_cP_2pi_phi__9 q[0], q[1];
_cP_2pi_phi__10 q[0], q[1];
if (c[0]) {
  p(-pi/2) q[0];
}
h q[0];
c[1] = measure q[0];
reset q[0];
h q[0];
_cP_2pi_phi__11 q[0], q[1];
_cP_2pi_phi__12 q[0], q[1];
if (c[0]) {
  p(-pi/4) q[0];
}
if (c[1]) {
  p(-pi/2) q[0];
}
h q[0];
c[2] = measure q[0];
reset q[0];
h q[0];
_cP_2pi_phi__13 q[0], q[1];
if (c[0]) {
  p(-pi/8) q[0];
}
if (c[1]) {
  p(-pi/4) q[0];
}
if (c[2]) {
  p(-pi/2) q[0];
}
h q[0];
c[3] = measure q[0];
