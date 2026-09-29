"""Sweep the AirwayEvaluation CLI over deformations x trees x hyperparameters.

For a planar and a non-planar 6-generation phantom, every deformation in
DEFORMATIONS is applied once (moving masks saved under OUT/masks), then the
CLI scores each moving mask:
    baseline   no registration, once per sensitivity multiplier
    deformable --deformableEvaluation for every combination in GRID
    threshold  default registration, other --branchDetectionThresholdPercent
               values (scoring sensitivity only; never tuned)

Results stream to OUT/results.jsonl (resumable: finished jobs are skipped) and
are collected into OUT/results.csv. The CLI is the only thing that registers or
scores anything.

    python DeformationSweep.py --workers 20 --threads 3
    python DeformationSweep.py --quick            # tiny grid, for testing
"""

import argparse
import itertools
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd

import AirwayEvaluationCLI as cli
import Deformations as D
from SyntheticAirway import make_synthetic_airway

import itk

GENERATIONS = 6
IMAGE_SIZE = 300
BIFURCATION_ANGLE = 45.0
TREES = {"planar": 0.0, "non-planar": 45.0}  # name -> plane twist (degrees)

# registration hyperparameters, full factorial
GRID = {
    "maximum_pairing_distance": [2.5, 5.0, 10.0, 20.0, 40.0],
    "minimum_landmark_distance": [2.0, 5.0, 10.0],
    "sensitivity_multiplier": [1.0, 2.0, 3.0],
    "projection_neighbors": [1, 5, 15],
}
QUICK_GRID = {
    "maximum_pairing_distance": [5.0, 20.0],
    "minimum_landmark_distance": [5.0],
    "sensitivity_multiplier": [2.0],
    "projection_neighbors": [5],
}
# scoring only: how BD reacts to the detection threshold, at default registration
DETECTION_THRESHOLDS = [25.0, 75.0]

TRANSLATION_DIRECTION = np.array([0.6, -0.5, 0.62]) / np.linalg.norm([0.6, -0.5, 0.62])
ROTATION_AXIS = (1.0, 1.0, 1.0)


def deformations(phantom):
    """(family, level label, magnitude, deformation) for one phantom, mildest first per family."""
    items = [("identity", "none", 0.0, D.translation((0, 0, 0)))]
    for m in (5, 15, 30, 60):
        items.append(("translation", f"{m} vox", m, D.translation(m * TRANSLATION_DIRECTION)))
    for a in (5, 10, 20, 30, 45):
        items.append(("rotation", f"{a}°", a, D.rotation(phantom, ROTATION_AXIS, a)))
    for a, m in ((5, 10), (10, 20), (20, 30), (30, 40)):
        items.append(("rotation+translation", f"{a}° + {m} vox", a,
                      D.rotation(phantom, ROTATION_AXIS, a, m * TRANSLATION_DIRECTION)))
    for s in (0.02, 0.05, 0.10, 0.20):
        items.append(("affine (shear)", f"shear {s:g}", s,
                      D.affine(phantom, (10, -5, 5), 10 * TRANSLATION_DIRECTION, (s, s / 2, s / 2))))
    for m in (1, 2, 4, 6, 8, 12, 16):
        items.append(("nonrigid", f"{m} vox", m, D.bspline(phantom, m)))
    for cc in (0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30):
        items.append(("breath hold", f"{cc:.0%}", cc * 100, D.BreathHold(cc)))
    return items


def slug(text):
    return "".join(c if c.isalnum() or c in ".=+-" else "_" for c in str(text)).strip("_")


def parameter_tag(parameters):
    short = {"maximum_pairing_distance": "p", "minimum_landmark_distance": "m",
             "sensitivity_multiplier": "s", "projection_neighbors": "n",
             "branch_detection_threshold_percent": "t"}
    return "_".join(f"{short[k]}{v:g}" for k, v in sorted(parameters.items()))


def prepare_masks(out, quick):
    """Write the fixed and moving masks; returns the deformation table."""
    rows = []
    for tree_name, twist in TREES.items():
        tree = make_synthetic_airway(image_size=IMAGE_SIZE, generations=GENERATIONS,
                                     bifurcation_angle=BIFURCATION_ANGLE, plane_twist=twist)
        phantom = D.Phantom(tree, tree_name)
        mask_dir = os.path.join(out, "masks", slug(tree_name))
        os.makedirs(mask_dir, exist_ok=True)
        fixed_path = os.path.join(mask_dir, "fixed.nii.gz")
        phantom.write(fixed_path)

        items = deformations(phantom)
        if quick:
            items = [items[0], items[6], items[-5]]
        for index, (family, level, magnitude, deformation) in enumerate(items):
            moving_path = os.path.join(mask_dir, f"{index:02d}_{slug(family)}_{slug(level)}.nii.gz")
            moving, mean_disp, peak_disp = D.deform(phantom, deformation)
            itk.imwrite(moving, moving_path, compression=True)
            rows.append({
                "tree": tree_name, "deformation_id": f"{index:02d}", "family": family, "level": level,
                "magnitude": magnitude, "mean_disp": mean_disp, "peak_disp": peak_disp,
                "moving_voxels": int((itk.array_view_from_image(moving) > 0).sum()),
                "fixed_voxels": int(phantom.array.sum()),
                "fixed_path": fixed_path, "moving_path": moving_path,
            })
            print(f"{tree_name:>10} {family:>20} {level:>14}: peak {peak_disp:5.1f} vox", flush=True)
    table = pd.DataFrame(rows)
    table.to_csv(os.path.join(out, "deformations.csv"), index=False)
    return table


def jobs_for(deformation_table, grid):
    """Every CLI call of the sweep as (job id, deformation row, mode, parameters, deformable)."""
    combos = [dict(zip(grid, values)) for values in itertools.product(*grid.values())]
    sensitivities = sorted({c["sensitivity_multiplier"] for c in combos})
    defaults = {k: v for k, v in cli.DEFAULT_PARAMETERS.items() if k != "branch_detection_threshold_percent"}

    jobs = []
    for _, d in deformation_table.iterrows():
        prefix = f"{slug(d.tree)}/{d.deformation_id}"
        for s in sensitivities:
            parameters = {"sensitivity_multiplier": s}
            jobs.append((f"{prefix}/baseline_{parameter_tag(parameters)}", d, "baseline", parameters, False))
        for parameters in combos:
            jobs.append((f"{prefix}/deformable_{parameter_tag(parameters)}", d, "deformable", parameters, True))
        for threshold in DETECTION_THRESHOLDS:
            parameters = {**defaults, "branch_detection_threshold_percent": threshold}
            jobs.append((f"{prefix}/threshold_{parameter_tag(parameters)}", d, "threshold", parameters, True))
    return jobs


def run_job(out, job, threads):
    job_id, d, mode, parameters, deformable = job
    start = time.time()
    run = cli.run_cli(d.moving_path, d.fixed_path, os.path.join(out, "runs", job_id), deformable=deformable,
                      parameters=parameters, volumes=(), branch_origins=False, threads=threads)
    row = {
        "job_id": job_id, "tree": d.tree, "deformation_id": d.deformation_id, "family": d.family,
        "level": d.level, "magnitude": d.magnitude, "mean_disp": d.mean_disp, "peak_disp": d.peak_disp,
        "mode": mode, **{k: parameters.get(k, v) for k, v in cli.DEFAULT_PARAMETERS.items()},
        "ok": run.ok, "error": run.error, "seconds": time.time() - start,
    }
    if run.ok:
        row.update({
            "TLD": run.tree_length_detected, "BD": run.branch_detection, "DSC": run.metrics["DSC"],
            "branches_detected": int(run.metrics["BranchesDetected"]),
            "total_branches": int(run.metrics["TotalBranches"]),
        })
        for g, detection in run.generations.items():
            row[f"g{g}_detected"] = detection.branches_detected
            row[f"g{g}_total"] = detection.total_branches
            row[f"g{g}_BD"] = detection.branch_detection
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="sweep_results")
    parser.add_argument("--workers", type=int, default=16, help="CLI processes at once")
    parser.add_argument("--threads", type=int, default=4, help="ITK threads per CLI process")
    parser.add_argument("--quick", action="store_true", help="3 deformations per tree, 2 parameter sets")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    print("CLI:", cli.DEFAULT_EXECUTABLE, "built", time.ctime(os.path.getmtime(cli.DEFAULT_EXECUTABLE)), flush=True)
    deformation_table = prepare_masks(args.out, args.quick)
    jobs = jobs_for(deformation_table, QUICK_GRID if args.quick else GRID)

    results_path = os.path.join(args.out, "results.jsonl")
    done = set()
    if os.path.exists(results_path):
        with open(results_path) as f:
            done = {json.loads(line)["job_id"] for line in f if line.strip()}
    todo = [job for job in jobs if job[0] not in done]
    print(f"{len(jobs)} jobs, {len(done)} already done, {len(todo)} to run "
          f"({args.workers} workers x {args.threads} threads)", flush=True)

    start = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as pool, open(results_path, "a") as results:
        futures = [pool.submit(run_job, args.out, job, args.threads) for job in todo]
        for count, future in enumerate(as_completed(futures), 1):
            results.write(json.dumps(future.result(), default=float) + "\n")
            results.flush()
            if count % 100 == 0 or count == len(todo):
                elapsed = time.time() - start
                print(f"{count}/{len(todo)} done, {elapsed / 60:.1f} min, "
                      f"~{elapsed / count * (len(todo) - count) / 60:.0f} min left", flush=True)

    with open(results_path) as f:
        table = pd.DataFrame([json.loads(line) for line in f if line.strip()])
    table.to_csv(os.path.join(args.out, "results.csv"), index=False)
    print(f"wrote {len(table)} rows to {os.path.join(args.out, 'results.csv')} "
          f"({(~table['ok']).sum()} CLI errors)", flush=True)


if __name__ == "__main__":
    main()
