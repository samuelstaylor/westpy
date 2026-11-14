"""Run 2: tracking ON, test mode, 3 BFGS cycles."""
from westpy import *

qedir = '/global/homes/s/sstaylor/software/QEdir_cpu/bin'
run_pw = f'srun -n 16 {qedir}/pw.x'
run_wbse = f'srun -n 16 {qedir}/wbse.x'
run_westpp = f'srun -n 16 {qedir}/westpp.x'
pp = [f'{element}_ONCV_PBE-1.2.upf' for element in ['C', 'H', 'O']]

bfgs = bfgs_iter_tracked(run_pw=run_pw, run_wbse=run_wbse, pp=pp, maxiter=3,
                         l_track_state=True,
                         run_westpp=run_westpp,
                         westpp_input='westpp.in',
                         l_test_mode=True)
bfgs.solve()
