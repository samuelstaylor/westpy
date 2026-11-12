"""NV- in MgO excited-state relaxation with state tracking (bfgs.py rewritten
for bfgs_iter_tracked). Same commands as the original bfgs.py, plus westpp.x.

Needs, in the run directory: pw.in, nscf.in, wbse_init.in, wbse.in, westpp.in,
and the pseudopotentials. See westpp_tracking.in next to this file for a
westpp input, and STATE_TRACKING.md for what ends up in tracking/.
"""
from westpy import *

qedir = '/global/homes/s/sstaylor/software/QEdir/bin'

run_pw = f'srun {qedir}/pw.x -nk 2'
pw_input = 'pw.in'
run_nscf = f'srun {qedir}/pw.x -nk 2'
nscf_input = 'nscf.in'
run_wbse_init = f'srun {qedir}/wbse_init.x -nb 16'
wbse_init_input = 'wbse_init.in'
run_wbse = f'srun {qedir}/wbse.x -nb 16'
wbse_input = 'wbse.in'
run_westpp = f'srun -n 16 {qedir}/westpp.x'  # as in job_submit.sh

pp = [f'{element}_ONCV_PBE-1.2.upf' for element in ['Mg', 'N', 'O']]

bfgs = bfgs_iter_tracked(run_pw=run_pw, pw_input=pw_input,
                         run_nscf=run_nscf, nscf_input=nscf_input,
                         run_wbse_init=run_wbse_init, wbse_init_input=wbse_init_input,
                         run_wbse=run_wbse, wbse_input=wbse_input,
                         pp=pp,
                         l_restart=True,
                         # --- state tracking ---
                         l_track_state=True,
                         run_westpp=run_westpp,
                         westpp_input='westpp.in',
                         l_track_orbitals=True,
                         band_window=None,        # automatic; or e.g. (539, 900)
                         overlap_thr=0.5,
                         ambiguity_thr=0.1,
                         stop_on_low_overlap=False,
                         l_test_mode=False)       # True keeps every step (lots of disk)
bfgs.solve()
