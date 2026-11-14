"""Check the formaldehyde smoke-test runs (read-only).

    python check_outputs.py [BASE]    (default $SCRATCH/tracking_tests/formaldehyde)

Run 1 (tracking off) vs run 0 (plain bfgs_iter): same BFGS log numbers, same
trajectory, same set of output files. Run 2 (tracking on, test mode): the
tracking/ and tracking/test/ contents.
"""

import glob
import json
import os
import re
import sys

import numpy as np

BASE = sys.argv[1] if len(sys.argv) > 1 else os.path.expandvars("$SCRATCH/tracking_tests/formaldehyde")
TOL = 1e-8  # Ry; same binary, same layout: expected to agree to (nearly) all digits
ok = True


def report(cond, msg):
    global ok
    ok &= bool(cond)
    print(("  PASS  " if cond else "  FAIL  ") + msg)


def bfgs_numbers(path):
    """(key, value) pairs of every number bfgs_iter prints, in order."""
    out = []
    for line in open(path):
        m = re.match(r"\s*(number of scf cycles|number of bfgs steps|energy old|energy new|total force|"
                     r"energy error|gradient error|new trust radius)\s*=\s*(\S+)", line)
        if m:
            out.append((m.group(1), float(m.group(2))))
    return out


def traj(path):
    text = open(path).read()
    energies = [float(x) for x in re.findall(r"(?<![\w])energy=(\S+)", text)]
    blocks, lines, i = [], text.splitlines(), 0
    while i < len(lines):
        if not lines[i].strip():
            i += 1
            continue
        nat = int(lines[i].split()[0])
        blocks.append(np.array([l.split()[1:4] for l in lines[i + 2:i + 2 + nat]], float))
        i += nat + 2
    return energies, blocks


def files(d):
    skip = re.compile(r"^(slurm_\d+\.(out|err)|bfgs\.out|job_.*\.sh)$")  # job scripts are named per run
    return sorted(os.path.relpath(p, d) for p in glob.glob(d + "/**", recursive=True)
                  if os.path.isfile(p) and not skip.match(os.path.basename(p)))


print(f"== base: {BASE}")
r0, r1, r2 = (os.path.join(BASE, r) for r in ("0_plain", "1_off", "2_on"))

print("\n== run 1 (tracking off) vs run 0 (plain bfgs_iter)")
if all(os.path.exists(os.path.join(r, "bfgs.out")) for r in (r0, r1)):
    n0, n1 = bfgs_numbers(r0 + "/bfgs.out"), bfgs_numbers(r1 + "/bfgs.out")
    report(len(n0) > 0 and [k for k, _ in n0] == [k for k, _ in n1], f"same BFGS log lines ({len(n0)} numbers)")
    if n0 and len(n0) == len(n1):
        dmax = max(abs(a - b) for (_, a), (_, b) in zip(n0, n1))
        report(dmax < TOL, f"max |difference| of logged numbers = {dmax:.3e} (tol {TOL:g})")
    e0, p0 = traj(r0 + "/bfgs_traj.xyz")
    e1, p1 = traj(r1 + "/bfgs_traj.xyz")
    report(len(e0) == len(e1), f"trajectory frames: {len(e0)} vs {len(e1)}")
    if len(e0) == len(e1):
        report(max(abs(a - b) for a, b in zip(e0, e1)) < TOL, "trajectory energies agree")
        report(max(np.abs(a - b).max() for a, b in zip(p0, p1)) < 1e-8, "trajectory positions agree")
    f0, f1 = files(r0), files(r1)
    report(f0 == f1, f"same output files ({len(f0)})" + ("" if f0 == f1 else f": only0={sorted(set(f0)-set(f1))} only1={sorted(set(f1)-set(f0))}"))
    report(not os.path.exists(r1 + "/tracking") and not os.path.exists(r1 + "/wbse.in.orig")
           and not os.path.exists(r1 + "/bfgs_tracking.json"), "run 1 wrote no tracking files")
    for line in open(r0 + "/bfgs.out"):
        if "Final energy" in line or "maximum number" in line or "converged" in line:
            print("        run 0:", line.strip())
else:
    print("  (runs 0/1 not finished)")

print("\n== run 2 (tracking on, test mode, maxiter=3)")
if os.path.exists(r2 + "/bfgs.out"):
    log = open(r2 + "/bfgs.out").read()
    warns = [l.strip() for l in log.splitlines() if "WARNING (tracking)" in l]
    print("  startup/step warnings:", *(warns or ["none"]), sep="\n        ")
    hist = json.load(open(r2 + "/tracking/history.json"))
    report(len(hist) == 3, f"history has {len(hist)} steps")
    report(sorted(os.path.basename(p) for p in glob.glob(r2 + "/tracking/test/step-*")) == ["step-1", "step-2", "step-3"],
           "tracking/test/step-1..3 exist")
    for n in (1, 2, 3):
        t = f"{r2}/tracking/test/step-{n}"
        need = ["tracking_summary.txt", "wbse.out", "westpp.out", "pwscf.save/wfc1.dat", "west.westpp.save"]
        missing = [f for f in need if not os.path.exists(os.path.join(t, f))]
        report(not missing, f"step-{n} contents" + (f" missing {missing}" if missing else ""))
    print("  swap_summary.txt:")
    print("".join("        " + l for l in open(r2 + "/tracking/test/swap_summary.txt")))
    for r in hist:
        row = r["overlap_row"]
        row_s = "  ".join(f"{j}:{o:.4f}" for j, o in row.items()) if row else "(reference)"
        print(f"  step {r['scf_iter']}: {r['label_before']} -> {r['label_after']}  swap={r['swap']}  "
              f"O = {row_s}  windows={r['windows']}  retained={r['retained_weight_min']:.4f}")
        for fs, v, ts, c, x in r.get("dominant_transitions", [])[:3]:
            print(f"        {v} -> {c}  X = {x:+.4f}")
    for r in hist[1:]:
        report(r["max_overlap"] is not None and r["max_overlap"] > 0.9, f"step {r['scf_iter']}: max overlap {r['max_overlap']:.4f} > 0.9")
else:
    print("  (run 2 not finished)")

print("\nALL CHECKS PASSED" if ok else "\nSOME CHECKS FAILED")
