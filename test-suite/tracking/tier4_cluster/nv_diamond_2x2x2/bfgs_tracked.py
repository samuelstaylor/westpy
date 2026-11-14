"""NV- in diamond 2x2x2 (63 atoms): tracked 3E relaxation from the C3v
ground-state geometry, forces_state 1, test mode. CPU build, 48 MPI tasks.
Restarts automatically when bfgs_tmp.json exists (resubmit job_relax.sh)."""
import os
from westpy import *

qedir = '/global/homes/s/sstaylor/software/QEdir_cpu/bin'
run_pw = f'srun -n 48 {qedir}/pw.x'
run_wbse = f'srun -n 48 {qedir}/wbse.x'
run_westpp = f'srun -n 48 {qedir}/westpp.x'
pp = [f'{element}_ONCV_PBE-1.2.upf' for element in ['C', 'N']]

bfgs = bfgs_iter_tracked(run_pw=run_pw, run_wbse=run_wbse, pp=pp,
                         maxiter=20,
                         l_restart=os.path.exists('bfgs_tmp.json'),
                         l_track_state=True,
                         run_westpp=run_westpp,
                         westpp_input='westpp.in',
                         l_test_mode=True)
bfgs.solve()
