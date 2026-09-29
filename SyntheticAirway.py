"""Synthetic bifurcating airway phantoms for testing the deformable pipeline.

Usage:
    from SyntheticAirway import SyntheticTube, make_synthetic_pair

    fixed, moving = make_synthetic_pair(image_size=300)
    fixed_image = fixed.to_itk()
"""

import os
import sys

import numpy as np

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
    "SyntheticTube",
    "DEFAULT_TREE_PARAMETERS",
    "make_synthetic_airway",
    "make_synthetic_pair",
    "write_generation_series",
    "make_single_tube",
    "make_y",
    "write_simple_shapes",
]


class SyntheticTube:
    def __init__(self, size=(128, 128, 128)):
        self.size = size
        self.mask = np.zeros(size, dtype=np.uint8)
        self.grid = np.indices(size, dtype=np.int32).transpose(1, 2, 3, 0)
        self.branches = []
        self.pruned = []

    @staticmethod
    def normalize(v):
        return v / np.linalg.norm(v)

    @staticmethod
    def rotate(v, axis, angle):
        axis = SyntheticTube.normalize(axis)
        return (
            v * np.cos(angle)
            + np.cross(axis, v) * np.sin(angle)
            + axis * np.dot(axis, v) * (1 - np.cos(angle))
        )

    @staticmethod
    def point_segment_distance(p, a, b):
        d = b - a
        n = np.dot(d, d)

        if n == 0:
            return np.linalg.norm(p - a)

        t = np.clip(np.dot(p - a, d) / n, 0, 1)
        return np.linalg.norm(p - (a + t * d))

    def add_tube(
        self,
        centerline,
        radius,
        generation,
        branch_id,
        parent_id,
        flat_ends=False,
    ):
        """Rasterize a tube of the given radius around centerline.

        By default the ends are hemispherical caps; flat_ends=True cuts them
        off perpendicular to the centerline so the tube is exactly as long
        as its centerline.
        """
        centerline = np.asarray(centerline, float)
        last = len(centerline) - 2

        for i, (p0, p1) in enumerate(zip(centerline[:-1], centerline[1:])):
            d = p1 - p0
            n = np.dot(d, d)

            if n == 0:
                continue

            lo = np.clip(
                np.floor(np.minimum(p0, p1) - radius).astype(int),
                0,
                self.size,
            )

            hi = np.clip(
                np.ceil(np.maximum(p0, p1) + radius).astype(int) + 1,
                0,
                self.size,
            )

            if np.any(hi <= lo):
                continue

            region = tuple(slice(a, b) for a, b in zip(lo, hi))
            points = self.grid[region]

            t_raw = np.sum((points - p0) * d, axis=-1) / n
            t = np.clip(t_raw, 0, 1)

            distance = np.linalg.norm(
                points - (p0 + t[..., None] * d),
                axis=-1,
            )

            inside = distance <= radius

            if flat_ends:
                # Only trim at the tube's two ends; interior joints keep
                # their rounding so bends stay filled. The end is exclusive
                # so an integer-length tube covers exactly that many voxels.
                if i == 0:
                    inside &= t_raw >= 0
                if i == last:
                    inside &= t_raw < 1

            self.mask[region][inside] = 1

        self.branches.append({
            "branch_id": branch_id,
            "parent_id": parent_id,
            "generation": generation,
            "centerline": centerline,
            "radius": radius,
            "length": np.linalg.norm(
                np.diff(centerline, axis=0),
                axis=1,
            ).sum(),
        })

    def clear_length(
        self,
        origin,
        direction,
        radius,
        length,
        ignore,
        min_gap,
        step=0.25,
    ):
        for s in np.arange(0, length + step, step):
            s = min(s, length)
            p = origin + s * direction

            for branch in self.branches:
                if branch["branch_id"] in ignore:
                    continue

                clearance = radius + branch["radius"] + min_gap
                c = branch["centerline"]

                if any(
                    self.point_segment_distance(p, a, b) < clearance
                    for a, b in zip(c[:-1], c[1:])
                ):
                    return max(0, s - step)

        return length

    def grow(
        self,
        centerline,
        radius,
        generation,
        branch_id,
        parent_id,
        max_generations,
        branch_length,
        radius_scale,
        length_scale,
        angle,
        normal,
        min_gap,
        min_length_fraction,
        pruned_length_scale,
        exact_length,
        plane_twist=0.0,
    ):
        self.add_tube(
            centerline,
            radius,
            generation,
            branch_id,
            parent_id,
        )

        if generation >= max_generations - 1:
            return

        direction = self.normalize(
            centerline[-1] - centerline[-2]
        )

        origin = centerline[-1]

        directions = [
            self.normalize(self.rotate(direction, normal, angle)),
            self.normalize(self.rotate(direction, normal, -angle)),
        ]

        child_radius = radius * radius_scale
        child_length = branch_length * length_scale
        child_ids = (2 * branch_id + 1, 2 * branch_id + 2)

        for child_id, child_direction in zip(child_ids, directions):

            if exact_length:
                # Every surviving branch is exactly child_length long, so
                # branch length depends only on generation (and not at all
                # when length_scale=1).
                requested_length = child_length
                min_length = child_length
            else:
                # Give the candidate branch extra room to extend before
                # deciding whether it is pruned.
                requested_length = child_length * pruned_length_scale
                min_length = min_length_fraction * child_length

            length = self.clear_length(
                origin,
                child_direction,
                child_radius,
                requested_length,
                {branch_id, *child_ids},
                min_gap,
            )

            if length < min_length:
                self.pruned.append(child_id)
                continue

            child = np.array([
                origin,
                origin + child_direction * length / 2,
                origin + child_direction * length,
            ])

            self.grow(
                child,
                child_radius,
                generation + 1,
                child_id,
                branch_id,
                max_generations,
                child_length,
                radius_scale,
                length_scale,
                angle,
                # plane_twist=0 keeps the same bifurcation plane every
                # generation (planar tree); otherwise rotate it about the child
                self.normalize(self.rotate(normal, child_direction, plane_twist)),
                min_gap,
                min_length_fraction,
                pruned_length_scale,
                exact_length,
                plane_twist,
            )

    def make_tree(
        self,
        centerline,
        radius=6,
        generations=5,
        branch_length=15,
        radius_scale=0.75,
        length_scale=0.75,
        bifurcation_angle=45,
        plane_normal=(1, 0, 0),
        min_gap=1,
        min_length_fraction=0.5,
        pruned_length_scale=1.5,
        exact_length=False,
        plane_twist=0.0,
    ):
        """Grow a bifurcating tree from the end of centerline.

        Child branches are length_scale times their parent's length. By
        default each branch may stretch up to pruned_length_scale times that
        or shrink to min_length_fraction of it to avoid collisions. With
        exact_length=True every branch is exactly its nominal length (pruned
        if it does not fit), so length_scale=1 gives identical branch lengths
        across all generations.

        plane_twist (degrees) rotates each child's bifurcation plane about the
        child's direction; 0 gives a planar tree, 90 alternates planes like a
        real airway.
        """
        centerline = np.asarray(centerline, float)

        direction = self.normalize(
            centerline[-1] - centerline[-2]
        )

        normal = np.asarray(plane_normal, float)
        normal -= np.dot(normal, direction) * direction

        if np.linalg.norm(normal) < 1e-6:
            raise ValueError(
                "plane_normal is parallel to root direction"
            )

        self.grow(
            centerline,
            radius,
            0,
            0,
            None,
            generations,
            branch_length,
            radius_scale,
            length_scale,
            np.deg2rad(bifurcation_angle),
            self.normalize(normal),
            min_gap,
            min_length_fraction,
            pruned_length_scale,
            exact_length,
            np.deg2rad(plane_twist),
        )

        return self

    def to_numpy(self):
        return self.mask

    def to_itk(self):
        return itk.image_from_array(self.mask)


# Tree parameters used in SyntheticTesting.ipynb.
DEFAULT_TREE_PARAMETERS = dict(
    radius=6,
    generations=7,
    branch_length=30,
    bifurcation_angle=45.0,
    plane_normal=(1.0, 0.0, 0.0),
)


def make_synthetic_airway(
    image_size=300,
    trachea_xy=(128, 128),
    trachea_z=(40, 70, 100),
    **tree_parameters,
):
    """Build one synthetic airway tree whose trachea runs along z at trachea_xy.

    Any SyntheticTube.make_tree keyword overrides DEFAULT_TREE_PARAMETERS.
    """
    parameters = {**DEFAULT_TREE_PARAMETERS, **tree_parameters}

    airway = SyntheticTube(size=(image_size, image_size, image_size))
    airway.make_tree(
        centerline=[(trachea_xy[0], trachea_xy[1], z) for z in trachea_z],
        **parameters,
    )
    return airway


def make_synthetic_pair(
    image_size=300,
    fixed_xy=(128, 128),
    moving_xy=(175, 175),
    **tree_parameters,
):
    """Build the (fixed, moving) pair from SyntheticTesting.ipynb: identical
    trees whose tracheas are offset in x/y."""
    fixed = make_synthetic_airway(image_size, fixed_xy, **tree_parameters)
    moving = make_synthetic_airway(image_size, moving_xy, **tree_parameters)
    return fixed, moving


def write_generation_series(
    out_dir,
    generations=range(11),
    image_size=300,
    **tree_parameters,
):
    """Write one NIfTI mask per generation count. Returns the written paths."""
    os.makedirs(out_dir, exist_ok=True)
    paths = []

    for n in generations:
        airway = make_synthetic_airway(
            image_size, generations=n, **tree_parameters
        )
        path = os.path.join(out_dir, f"synthetic_airway_gen{n:02d}.nii.gz")
        itk.imwrite(airway.to_itk(), path, compression=True)
        print(
            f"generations={n}: {len(airway.branches)} branches, "
            f"{len(airway.pruned)} pruned -> {path}",
            flush=True,
        )
        paths.append(path)

    return paths


def make_single_tube(length, image_size=300, radius=6, center_xy=(128, 128)):
    """One straight tube of the given length along z, centered in z."""
    center_z = image_size / 2
    airway = SyntheticTube(size=(image_size, image_size, image_size))
    airway.add_tube(
        [
            (center_xy[0], center_xy[1], center_z - length / 2),
            (center_xy[0], center_xy[1], center_z),
            (center_xy[0], center_xy[1], center_z + length / 2),
        ],
        radius,
        generation=0,
        branch_id=0,
        parent_id=None,
        flat_ends=True,
    )
    return airway


def make_y(length, image_size=300, radius=6, center_xy=(128, 128), **tree_parameters):
    """A trachea of the given length splitting into two arms of the same length."""
    start_z = image_size / 2 - length
    parameters = dict(
        radius=radius,
        generations=2,
        branch_length=length,
        length_scale=1.0,
        exact_length=True,
        **tree_parameters,
    )
    return make_synthetic_airway(
        image_size,
        center_xy,
        (start_z, start_z + length / 2, start_z + length),
        **parameters,
    )


def write_simple_shapes(out_dir, length=90, image_size=300, radius=6):
    """Write a Y and single tubes of length, length/2 and length/3 as NIfTI.
    Returns the written paths."""
    os.makedirs(out_dir, exist_ok=True)
    shapes = {
        "synthetic_y": make_y(length, image_size, radius),
        "synthetic_tube_L": make_single_tube(length, image_size, radius),
        "synthetic_tube_L_div2": make_single_tube(length / 2, image_size, radius),
        "synthetic_tube_L_div3": make_single_tube(length / 3, image_size, radius),
    }
    paths = []

    for name, airway in shapes.items():
        path = os.path.join(out_dir, f"{name}.nii.gz")
        itk.imwrite(airway.to_itk(), path, compression=True)
        lengths = ", ".join(f"{b['length']:.1f}" for b in airway.branches)
        print(f"{name}: branch lengths [{lengths}] -> {path}", flush=True)
        paths.append(path)

    return paths


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Write synthetic airway NIfTI masks for a range of generations."
    )
    parser.add_argument("--out-dir", default="synthetic_nifti")
    parser.add_argument("--max-generations", type=int, default=10)
    parser.add_argument("--image-size", type=int, default=300)
    parser.add_argument(
        "--simple-shapes",
        action="store_true",
        help="Write a Y and single tubes of length L, L/2, L/3 instead of "
        "the generation series.",
    )
    parser.add_argument(
        "--length", type=float, default=90, help="L for --simple-shapes."
    )
    parser.add_argument(
        "--identical-lengths",
        action="store_true",
        help="Give every branch the same length regardless of generation "
        "(length_scale=1, exact_length=True).",
    )
    args = parser.parse_args()

    if args.simple_shapes:
        write_simple_shapes(args.out_dir, args.length, args.image_size)
        sys.exit()

    tree_parameters = {}
    if args.identical_lengths:
        tree_parameters = dict(length_scale=1.0, exact_length=True)

    write_generation_series(
        args.out_dir,
        range(args.max_generations + 1),
        args.image_size,
        **tree_parameters,
    )
