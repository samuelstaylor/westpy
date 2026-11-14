"""Run 2: tracking ON, test mode, relaxed to convergence (bfgs_iter default
maxiter = 100). Restarts automatically if bfgs_tmp.json exists, so a run cut
off by the debug walltime can be continued by resubmitting job_2_on.sh."""
import os
from westpy import *

qedir = '/global/homes/s/sstaylor/software/QEdir_cpu/bin'
run_pw = f'srun -n 16 {qedir}/pw.x'
run_wbse = f'srun -n 16 {qedir}/wbse.x'
run_westpp = f'srun -n 16 {qedir}/westpp.x'
pp = [f'{element}_ONCV_PBE-1.2.upf' for element in ['C', 'H', 'O']]

bfgs = bfgs_iter_tracked(run_pw=run_pw, run_wbse=run_wbse, pp=pp,
                         l_restart=os.path.exists('bfgs_tmp.json'),
                         l_track_state=True,
                         run_westpp=run_westpp,
                         westpp_input='westpp.in',
                         l_test_mode=True)
bfgs.solve()
