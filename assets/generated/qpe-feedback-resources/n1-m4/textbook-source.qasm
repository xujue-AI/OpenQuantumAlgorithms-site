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
qubit[5] q;
reset q[0];
reset q[1];
reset q[2];
reset q[3];
reset q[4];
x q[4];
h q[0];
h q[1];
h q[2];
h q[3];
_cP_2pi_phi_ q[0], q[4];
_cP_2pi_phi__0 q[1], q[4];
_cP_2pi_phi__1 q[1], q[4];
_cP_2pi_phi__2 q[2], q[4];
_cP_2pi_phi__3 q[2], q[4];
_cP_2pi_phi__4 q[2], q[4];
_cP_2pi_phi__5 q[2], q[4];
_cP_2pi_phi__6 q[3], q[4];
_cP_2pi_phi__7 q[3], q[4];
_cP_2pi_phi__8 q[3], q[4];
_cP_2pi_phi__9 q[3], q[4];
_cP_2pi_phi__10 q[3], q[4];
_cP_2pi_phi__11 q[3], q[4];
_cP_2pi_phi__12 q[3], q[4];
_cP_2pi_phi__13 q[3], q[4];
swap q[1], q[2];
swap q[0], q[3];
h q[0];
cp(-pi/2) q[0], q[1];
h q[1];
cp(-pi/4) q[0], q[2];
cp(-pi/2) q[1], q[2];
h q[2];
cp(-pi/8) q[0], q[3];
cp(-pi/4) q[1], q[3];
cp(-pi/2) q[2], q[3];
h q[3];
c[0] = measure q[0];
c[1] = measure q[1];
c[2] = measure q[2];
c[3] = measure q[3];
