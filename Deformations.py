"""Known deformations of a synthetic airway, for stress-testing the registration.

Every ITK transform here is a resampling transform: moving(x) = fixed(T(x)).
Masks are warped with linear interpolation and re-thresholded at 0.5 so shear
and nonrigid warps stay smooth. BreathHold instead rebuilds the tree from its
centerlines (no resampling).

Usage:
    from SyntheticAirway import make_synthetic_airway
    import Deformations as D

    phantom = D.Phantom(make_synthetic_airway(generations=6, plane_twist=45), "non-planar")
    moving, mean_disp, peak_disp = D.deform(phantom, D.rotation(phantom, (1, 1, 0), 10))
"""

import numpy as np

import SyntheticAirway  # puts the custom ITK build on sys.path before itk
from SyntheticAirway import SyntheticTube

import itk

__all__ = [
    "Phantom",
    "translation",
    "rigid",
    "rotation",
    "affine",
    "bspline",
    "BreathHold",
    "displacement_stats",
    "warp",
    "deform",
]


class Phantom:
    """A fixed airway plus what the deformations need: its mask, a float copy
    for interpolation, its centroid, and a sample of airway voxels."""

    def __init__(self, tree, name, seed=0, sample_size=4000):
        self.tree = tree
        self.name = name
        self.mask = tree.to_itk()
        self.array = tree.to_numpy() > 0
        self.image_size = tree.size[0]

        self.float_image = itk.image_from_array(self.array.astype(np.float32))
        self.float_image.CopyInformation(self.mask)

        # rotation / shear center: airway centroid in ITK physical (x, y, z) order
        # (numpy is [z, y, x]; spacing 1, origin 0)
        self.center = np.argwhere(self.array).mean(axis=0)[::-1].tolist()

        # airway voxels (ITK x, y, z) used to measure / scale displacements
        points = np.argwhere(self.array)[:, ::-1].astype(float)
        rng = np.random.default_rng(seed)
        self.sample_points = points[rng.choice(len(points), min(sample_size, len(points)), replace=False)]

    def write(self, path):
        itk.imwrite(self.mask, path, compression=True)


# -------------------------------------------------------------------------
# Transforms
# -------------------------------------------------------------------------

def translation(offset):
    T = itk.TranslationTransform[itk.D, 3].New()
    T.SetOffset([float(v) for v in offset])
    return T


def _euler_matrix(angles_deg):
    E = itk.Euler3DTransform[itk.D].New()
    E.SetRotation(*np.deg2rad(angles_deg))
    return itk.array_from_matrix(E.GetMatrix())


def rigid(phantom, angles_deg, offset=(0, 0, 0)):
    """Euler angles (degrees about x, y, z) about the airway centroid, then offset."""
    E = itk.Euler3DTransform[itk.D].New()
    E.SetCenter(phantom.center)
    E.SetRotation(*np.deg2rad(angles_deg))
    E.SetTranslation([float(v) for v in offset])
    return E


def rotation(phantom, axis, angle_deg, offset=(0, 0, 0)):
    """Rotation by angle_deg about axis (ITK x, y, z) through the airway centroid, then offset."""
    axis = np.asarray(axis, float)
    V = itk.VersorRigid3DTransform[itk.D].New()
    V.SetCenter(phantom.center)
    V.SetRotation(itk.Vector[itk.D, 3]((axis / np.linalg.norm(axis)).tolist()), float(np.deg2rad(angle_deg)))
    V.SetTranslation([float(v) for v in offset])
    return V


def affine(phantom, angles_deg, offset, shear):
    """Rotation @ shear about the centroid. shear is the (xy, xz, yz) off-diagonals."""
    S = np.eye(3)
    S[0, 1], S[0, 2], S[1, 2] = shear
    A = itk.AffineTransform[itk.D, 3].New()
    A.SetCenter(phantom.center)
    A.SetMatrix(itk.matrix_from_array(_euler_matrix(angles_deg) @ S))
    A.SetTranslation([float(v) for v in offset])
    return A


def displacement_stats(phantom, transform):
    """(mean, max) displacement of the sampled airway voxels, in voxels."""
    moved = np.array([transform.TransformPoint(p.tolist()) for p in phantom.sample_points])
    d = np.linalg.norm(moved - phantom.sample_points, axis=1)
    return d.mean(), d.max()


def bspline(phantom, peak_displacement, mesh_size=6, seed=0):
    """Random smooth B-spline warp scaled so the largest displacement of any
    sampled airway voxel is peak_displacement voxels (control spacing ~ image size / mesh_size)."""
    B = itk.BSplineTransform[itk.D, 3, 3].New()
    B.SetTransformDomainOrigin([0.0, 0.0, 0.0])
    B.SetTransformDomainPhysicalDimensions([phantom.image_size - 1.0] * 3)
    B.SetTransformDomainMeshSize([mesh_size] * 3)
    direction = itk.Matrix[itk.D, 3, 3]()
    direction.SetIdentity()
    B.SetTransformDomainDirection(direction)

    n = B.GetNumberOfParameters()
    coefficients = np.random.default_rng(seed).normal(size=n)

    def set_coefficients(values):
        parameters = itk.OptimizerParameters[itk.D](n)
        for i, v in enumerate(values):
            parameters.SetElement(i, float(v))
        B.SetParametersByValue(parameters)

    # displacement is linear in the coefficients, so one rescale hits the target
    set_coefficients(coefficients)
    _, peak = displacement_stats(phantom, B)
    set_coefficients(coefficients * peak_displacement / peak)
    return B


class BreathHold:
    """A second breath hold of a tree, rebuilt from its centerlines.

    Nonterminal branches follow a smooth breathing field anchored at the top
    of the trachea: craniocaudal stretch growing quadratically with depth (the
    diaphragm moves the bases most) plus lateral and anteroposterior expansion
    about the trachea axis growing linearly with depth. Terminal branches are
    carried as rigid bodies by their (moved) origins. Tubes are re-rasterized
    with their original radii, so the moving airway is not resampled.

    craniocaudal: displacement at the deepest point as a fraction of tree depth.
    lateral / anteroposterior: expansion fractions (default 0.5x / 0.3x craniocaudal).
    asymmetry: one side expands (1 + a)x, the other (1 - a)x laterally.
    shift: whole-patient shift between scans, ITK (x, y, z) voxels.
    """

    def __init__(self, craniocaudal, lateral=None, anteroposterior=None, asymmetry=0.0, shift=(0, 0, 0)):
        self.craniocaudal = craniocaudal
        self.lateral = 0.5 * craniocaudal if lateral is None else lateral
        self.anteroposterior = 0.3 * craniocaudal if anteroposterior is None else anteroposterior
        self.asymmetry = asymmetry
        self.shift = tuple(shift)

    @property
    def name(self):
        return (f"cc={self.craniocaudal:.0%}" + (" asym" if self.asymmetry else "")
                + (" +shift" if any(self.shift) else ""))

    def describe(self):
        text = f"cc={self.craniocaudal:.0%} lat={self.lateral:.0%} ap={self.anteroposterior:.0%}"
        if self.asymmetry:
            text += f" asym={self.asymmetry:g}"
        if any(self.shift):
            text += f" shift={self.shift}"
        return text

    def build(self, tree):
        """Returns (moved SyntheticTube, displacement magnitude of every centerline point).

        SyntheticTube coordinates are numpy [z, y, x] = ITK (z, y, x): the trachea
        runs along axis 2 (craniocaudal), branches spread along axis 1 (lateral),
        axis 0 is anteroposterior.
        """
        apex = tree.branches[0]["centerline"][0]
        depth = max(b["centerline"][:, 2].max() for b in tree.branches) - apex[2]
        shift = np.asarray(self.shift, float)[::-1]

        def displacement(points):
            d = np.clip(points[:, 2] - apex[2], 0, None) / depth
            side = np.where(points[:, 1] >= apex[1], 1 + self.asymmetry, 1 - self.asymmetry)
            u = np.zeros_like(points)
            u[:, 2] = self.craniocaudal * depth * d ** 2
            u[:, 1] = self.lateral * (points[:, 1] - apex[1]) * d * side
            u[:, 0] = self.anteroposterior * (points[:, 0] - apex[0]) * d
            return u + shift

        parents = {b["parent_id"] for b in tree.branches}
        moved = SyntheticTube(size=tree.size)
        magnitudes = []
        for branch in tree.branches:
            centerline = branch["centerline"]
            if branch["branch_id"] in parents:  # nonterminal: follows the breathing field
                u = displacement(centerline)
            else:  # terminal: rigid body carried by its origin
                u = np.repeat(displacement(centerline[:1]), len(centerline), axis=0)
            moved.add_tube(centerline + u, branch["radius"], branch["generation"],
                           branch["branch_id"], branch["parent_id"])
            magnitudes.append(np.linalg.norm(u, axis=1))
        return moved, np.concatenate(magnitudes)


# -------------------------------------------------------------------------
# Applying a deformation
# -------------------------------------------------------------------------

def warp(phantom, transform):
    """Resample the phantom mask through transform (linear, thresholded at 0.5)."""
    moving = itk.resample_image_filter(
        phantom.float_image,
        transform=transform,
        use_reference_image=True,
        reference_image=phantom.float_image,
        interpolator=itk.LinearInterpolateImageFunction.New(phantom.float_image),
        default_pixel_value=0,
    )
    output = itk.image_from_array((itk.array_from_image(moving) >= 0.5).astype(np.uint8))
    output.CopyInformation(phantom.mask)
    return output


def deform(phantom, deformation):
    """Returns (moving mask, mean displacement, peak displacement) for an ITK
    transform or a BreathHold."""
    if isinstance(deformation, BreathHold):
        moved_tree, magnitudes = deformation.build(phantom.tree)
        return moved_tree.to_itk(), magnitudes.mean(), magnitudes.max()
    mean, peak = displacement_stats(phantom, deformation)
    return warp(phantom, deformation), mean, peak
