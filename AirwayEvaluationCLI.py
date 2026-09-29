"""Run the C++ AirwayEvaluation CLI and read back its metrics and debug volumes.

The CLI is the source of truth: nothing here re-implements the pipeline.

Usage:
    import AirwayEvaluationCLI as cli

    run = cli.run_cli("moving.nii.gz", "fixed.nii.gz", "out/case", deformable=True,
                      parameters={"maximum_pairing_distance": 10.0})
    print(run.tree_length_detected, run.branch_detection, run.generations)
    overlap = run.volume("overlapComparison")  # 1 overlap, 2 FN, 3 FP
"""

import csv
import os
import subprocess
import sys
from dataclasses import dataclass, field

# The airway filters (SkeletonizeAirway, SkeletonLabeling, ...) only exist in
# the custom ITK build, so it must be on the path before itk is first imported.
ITK_PYTHON_PATH = os.environ.get(
    "AIRWAY_ITK_PYTHON_PATH",
    "/raid0/homes/jkitzmann/dev/ITK-build-binarythinning/Wrapping/Generators/Python",
)
if os.path.isdir(ITK_PYTHON_PATH) and ITK_PYTHON_PATH not in sys.path:
    sys.path.insert(0, ITK_PYTHON_PATH)

import itk  # noqa: E402

__all__ = [
    "DEFAULT_EXECUTABLE",
    "PARAMETER_FLAGS",
    "DEFAULT_PARAMETERS",
    "VOLUMES",
    "OVERLAP",
    "FALSE_NEGATIVE",
    "FALSE_POSITIVE",
    "GenerationDetection",
    "CLIRun",
    "run_cli",
]

DEFAULT_EXECUTABLE = os.environ.get(
    "AIRWAY_EVALUATION_CLI",
    "/IPLlinux/raid0/homes/jkitzmann/Research/PCCTAirwaySegmentation/build/bin/AirwayEvaluation",
)

# pipeline hyperparameters (AirwayEvaluationParameters) -> CLI flag
PARAMETER_FLAGS = {
    "sensitivity_multiplier": "--sensitivityMultiplier",
    "projection_neighbors": "--projectionNeighbors",
    "branch_detection_threshold_percent": "--branchDetectionThresholdPercent",
    "minimum_landmark_distance": "--minimumLandmarkDistance",
    "maximum_pairing_distance": "--maximumPairingDistance",
}

# the CLI's defaults (AirwayEvaluation.xml)
DEFAULT_PARAMETERS = {
    "sensitivity_multiplier": 2.0,
    "projection_neighbors": 5,
    "branch_detection_threshold_percent": 50.0,
    "minimum_landmark_distance": 5.0,
    "maximum_pairing_distance": 5.0,
}

# CLI flag -> file written in the output directory
VOLUMES = {
    "overlapComparison": "overlapComparison.nii.gz",
    "treeLengthComparison": "treeLengthComparison.nii.gz",
    "registeredPrediction": "registeredPrediction.nii.gz",
    "predictionSkeleton": "predictionSkeleton.nii.gz",
    "groundTruthSkeleton": "groundTruthSkeleton.nii.gz",
    "labeledPrediction": "labeledPrediction.nii.gz",
    "labeledGroundTruth": "labeledGroundTruth.nii.gz",
    "generationPrediction": "generationPrediction.nii.gz",
    "generationGroundTruth": "generationGroundTruth.nii.gz",
}

# overlapComparison labels (OverlapLabel in AirwayEvaluationPipeline.h)
OVERLAP, FALSE_NEGATIVE, FALSE_POSITIVE = 1, 2, 3


@dataclass
class GenerationDetection:
    branches_detected: int
    total_branches: int
    branch_detection: float  # percent


@dataclass
class CLIRun:
    """One CLI invocation. Metrics are percentages [0, 100] as the CLI writes them."""
    out_dir: str
    command: list
    returncode: int
    log: str  # stdout + stderr
    metrics: dict = field(default_factory=dict)  # results.csv Metric -> value
    generations: dict = field(default_factory=dict)  # generation -> GenerationDetection

    @property
    def ok(self):
        return self.returncode == 0 and bool(self.metrics)

    @property
    def error(self):
        """The most informative line of the log when the run failed."""
        if self.ok:
            return ""
        lines = [line.strip() for line in self.log.splitlines() if line.strip()]
        for key in ("ITK ERROR:", "Description:", "what():", "ERROR", "Exception"):
            for line in lines:
                if key in line:
                    return line.split(key, 1)[1].strip() or line
        return lines[-1] if lines else f"exit code {self.returncode}"

    @property
    def tree_length_detected(self):
        return self.metrics.get("TreeLengthDetected")

    @property
    def branch_detection(self):
        return self.metrics.get("BranchDetection")

    @property
    def branches(self):
        if not self.ok:
            return ""
        return f"{int(self.metrics['BranchesDetected'])}/{int(self.metrics['TotalBranches'])}"

    def path(self, name):
        return os.path.join(self.out_dir, VOLUMES.get(name, name))

    def volume(self, name):
        """Read one of the written debug volumes (a VOLUMES key)."""
        path = self.path(name)
        return itk.imread(path) if os.path.exists(path) else None


def _read_metrics(path):
    metrics, generations = {}, {}
    if not os.path.exists(path):
        return metrics, generations
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            metrics[row["Metric"]] = float(row["Value"])

    ids = sorted({
        int(key[len("Generation"):].split("_")[0]) for key in metrics if key.startswith("Generation")
    })
    for g in ids:
        generations[g] = GenerationDetection(
            int(metrics[f"Generation{g}_BranchesDetected"]),
            int(metrics[f"Generation{g}_TotalBranches"]),
            metrics[f"Generation{g}_BranchDetection"],
        )
    return metrics, generations


def run_cli(
    prediction_path,
    ground_truth_path,
    out_dir,
    deformable=True,
    parameters=None,
    volumes=tuple(VOLUMES),
    branch_origins=True,
    threads=None,
    executable=DEFAULT_EXECUTABLE,
):
    """Run AirwayEvaluation on two mask files; everything it writes goes to out_dir.

    parameters: PARAMETER_FLAGS keys -> values; anything left out keeps the CLI default.
    volumes: which debug volumes to request (VOLUMES keys). registeredPrediction is
    only written by a deformable run. Requesting no volumes and no branch origins
    skips the label projections, which is faster.
    threads: ITK threads for this process (None = ITK's default).
    The log is saved as out_dir/log.txt.
    """
    unknown = set(parameters or {}) - set(PARAMETER_FLAGS)
    if unknown:
        raise ValueError(f"unknown CLI parameters {sorted(unknown)}; expected {sorted(PARAMETER_FLAGS)}")
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "results.csv")
    if os.path.exists(csv_path):
        os.remove(csv_path)  # never read a stale result

    command = [
        executable,
        "--predictionPath", str(prediction_path),
        "--groundTruthPath", str(ground_truth_path),
        "--csvOutputPath", csv_path,
    ]
    if branch_origins:
        command += ["--branchOriginsPath", os.path.join(out_dir, "branchOrigins.csv")]
    for name, value in (parameters or {}).items():
        text = str(int(value)) if name == "projection_neighbors" else repr(float(value))
        command += [PARAMETER_FLAGS[name], text]
    if deformable:
        command.append("--deformableEvaluation")
    for name in volumes:
        if name == "registeredPrediction" and not deformable:
            continue
        command += [f"--{name}", os.path.join(out_dir, VOLUMES[name])]

    env = None
    if threads:
        env = {**os.environ, "ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS": str(threads)}
    completed = subprocess.run(command, capture_output=True, text=True, env=env)
    log = completed.stdout + completed.stderr
    with open(os.path.join(out_dir, "log.txt"), "w") as f:
        f.write(" ".join(command) + "\n\n" + log)

    metrics, generations = _read_metrics(csv_path) if completed.returncode == 0 else ({}, {})
    return CLIRun(
        out_dir=out_dir,
        command=command,
        returncode=completed.returncode,
        log=log,
        metrics=metrics,
        generations=generations,
    )
