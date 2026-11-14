"""Run 1: bfgs_iter_tracked with tracking OFF. Must reproduce run 0 exactly."""
from westpy import *

qedir = '/global/homes/s/sstaylor/software/QEdir_cpu/bin'
run_pw = f'srun -n 16 {qedir}/pw.x'
run_wbse = f'srun -n 16 {qedir}/wbse.x'
pp = [f'{element}_ONCV_PBE-1.2.upf' for element in ['C', 'H', 'O']]

bfgs = bfgs_iter_tracked(run_pw=run_pw, run_wbse=run_wbse, pp=pp, maxiter=5,
                         l_track_state=False)
bfgs.solve()
