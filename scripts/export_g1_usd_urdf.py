"""Export the training G1 USD to a portable URDF. Requires usd-core, numpy, scipy.

Run from the repository root: python scripts/export_g1_usd_urdf.py
"""

import ast
import math
from pathlib import Path
import re
import struct
import xml.etree.ElementTree as ET

import numpy as np
from pxr import Gf, Usd, UsdGeom, UsdPhysics, UsdShade
from scipy.spatial import ConvexHull


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "active_adaptation/assets/G1/waist_unlocked.usd"
OUTPUT = SOURCE.parent / "waist_unlocked_urdf"


def numbers(values):
    return " ".join(f"{float(v):.12g}" for v in values)


def rotation(quaternion):
    return np.array(Gf.Matrix3d(Gf.Quatd(quaternion))).T


def rpy(matrix):
    pitch = math.atan2(-matrix[2, 0], math.hypot(matrix[0, 0], matrix[1, 0]))
    if abs(math.cos(pitch)) < 1e-8:
        return [math.atan2(-matrix[1, 2], matrix[1, 1]), pitch, 0]
    return [math.atan2(matrix[2, 1], matrix[2, 2]), pitch,
            math.atan2(matrix[1, 0], matrix[0, 0])]


def origin(parent, position=(0, 0, 0), orient=np.eye(3)):
    ET.SubElement(parent, "origin", xyz=numbers(position), rpy=numbers(rpy(orient)))


def write_stl(path, triangles):
    """Binary STL in metres; triangles are already expressed in the link frame."""
    with path.open("wb") as stream:
        stream.write(b"Exported from training waist_unlocked.usd".ljust(80, b"\0"))
        stream.write(struct.pack("<I", len(triangles)))
        for tri in triangles:
            normal = np.cross(tri[1] - tri[0], tri[2] - tri[0])
            length = np.linalg.norm(normal)
            if length:
                normal /= length
            stream.write(struct.pack("<12fH", *normal, *tri.flatten(), 0))


def mesh_data(prim, transform):
    mesh = UsdGeom.Mesh(prim)
    points = np.asarray(mesh.GetPointsAttr().Get(), dtype=float)
    points = (transform @ np.column_stack((points, np.ones(len(points)))).T).T[:, :3]
    indices = list(mesh.GetFaceVertexIndicesAttr().Get())
    triangles, offset = [], 0
    for count in mesh.GetFaceVertexCountsAttr().Get():
        face = indices[offset:offset + count]
        for i in range(1, count - 1):
            triangles.append(points[[face[0], face[i], face[i + 1]]])
        offset += count
    if mesh.GetOrientationAttr().Get() == "leftHanded":
        triangles = [t[[0, 2, 1]] for t in triangles]
    if np.linalg.det(transform[:3, :3]) < 0:
        triangles = [t[[0, 2, 1]] for t in triangles]
    return points, triangles


def color(prim):
    material, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
    if material:
        for child in Usd.PrimRange(material.GetPrim()):
            if child.IsA(UsdShade.Shader):
                for name in ("inputs:diffuse_color_constant", "inputs:diffuseColor"):
                    value = child.GetAttribute(name).Get()
                    if value is not None:
                        return [*value, 1]
    return [0.7, 0.7, 0.7, 1]


def training_limits():
    tree = ast.parse((ROOT / "active_adaptation/assets/humanoid.py").read_text())
    call = next(n.value for n in tree.body if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == "G1_WAIST_UNLOCKED_CFG"
                        for t in n.targets))
    actuators = next(k.value for k in call.keywords if k.arg == "actuators")
    actuator = actuators.values[0]
    return {k.arg: ast.literal_eval(k.value) for k in actuator.keywords
            if k.arg in ("effort_limit", "velocity_limit")}


def export():
    stage = Usd.Stage.Open(str(SOURCE))
    assert UsdGeom.GetStageMetersPerUnit(stage) == 1.0
    assert UsdGeom.GetStageUpAxis(stage) == "Z"
    OUTPUT.mkdir(exist_ok=True)
    (OUTPUT / "meshes").mkdir(exist_ok=True)
    cache = UsdGeom.XformCache()
    bodies = [p for p in stage.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI)]
    robot = ET.Element("robot", name="g1_waist_unlocked")
    robot.append(ET.Comment("Generated from ../waist_unlocked.usd; see README.md for scope."))
    counts = {"links": 0, "joints": 0, "visuals": 0, "collisions": 0}
    for body in bodies:
        link = ET.SubElement(robot, "link", name=body.GetName())
        counts["links"] += 1
        mass = UsdPhysics.MassAPI(body)
        inertial = ET.SubElement(link, "inertial")
        origin(inertial, mass.GetCenterOfMassAttr().Get())
        ET.SubElement(inertial, "mass", value=numbers([mass.GetMassAttr().Get()]))
        rot = rotation(mass.GetPrincipalAxesAttr().Get())
        inertia = rot @ np.diag(mass.GetDiagonalInertiaAttr().Get()) @ rot.T
        ET.SubElement(inertial, "inertia", **{
            f"i{a}{b}": numbers([inertia[i, j]])
            for a, b, i, j in [("x", "x", 0, 0), ("x", "y", 0, 1),
                               ("x", "z", 0, 2), ("y", "y", 1, 1),
                               ("y", "z", 1, 2), ("z", "z", 2, 2)]})
        inverse = cache.GetLocalToWorldTransform(body).GetInverse()
        for prim in Usd.PrimRange(body):
            if prim == body or not prim.IsA(UsdGeom.Gprim):
                continue
            transform = np.array(cache.GetLocalToWorldTransform(prim) * inverse).T
            collision = (prim.HasAPI(UsdPhysics.CollisionAPI)
                         and UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get())
            visual = (UsdGeom.Imageable(prim).ComputePurpose() != "guide"
                      and UsdGeom.Imageable(prim).ComputeVisibility() != "invisible")
            for kind, enabled in [("visual", visual), ("collision", collision)]:
                if not enabled:
                    continue
                label = str(prim.GetPath()).removeprefix(str(body.GetPath()) + "/").replace("/", "_")
                element = ET.SubElement(link, kind, name=f"{body.GetName()}_{label}_{kind}")
                counts[kind + "s"] += 1
                geometry = ET.SubElement(element, "geometry")
                if prim.IsA(UsdGeom.Mesh):
                    points, triangles = mesh_data(prim, transform)
                    approximation = prim.GetAttribute("physics:approximation").Get()
                    if kind == "collision" and approximation == "boundingCube":
                        local_points = np.asarray(UsdGeom.Mesh(prim).GetPointsAttr().Get())
                        lower, upper = local_points.min(axis=0), local_points.max(axis=0)
                        center = (transform @ np.r_[(lower + upper) / 2, 1])[:3]
                        assert np.allclose(transform[:3, :3].T @ transform[:3, :3], np.eye(3), atol=1e-6)
                        origin(element, center, transform[:3, :3])
                        ET.SubElement(geometry, "box", size=numbers(upper - lower))
                    else:
                        if kind == "collision" and approximation == "convexHull":
                            hull = ConvexHull(np.unique(points, axis=0))
                            triangles = []
                            for face, equation in zip(hull.simplices, hull.equations):
                                tri = hull.points[face]
                                if np.dot(np.cross(tri[1] - tri[0], tri[2] - tri[0]), equation[:3]) < 0:
                                    tri = tri[[0, 2, 1]]
                                triangles.append(tri)
                        elif kind == "collision" and approximation not in (None, "none"):
                            raise ValueError(f"Unsupported collision approximation: {approximation}")
                        filename = f"meshes/{body.GetName()}_{label}_{kind}.stl"
                        write_stl(OUTPUT / filename, triangles)
                        origin(element)
                        ET.SubElement(geometry, "mesh", filename=filename)
                else:
                    assert np.allclose(transform[:3, :3].T @ transform[:3, :3], np.eye(3), atol=1e-6)
                    position, rot = transform[:3, 3], transform[:3, :3]
                    if prim.IsA(UsdGeom.Sphere):
                        origin(element, position, rot)
                        ET.SubElement(geometry, "sphere", radius=numbers([UsdGeom.Sphere(prim).GetRadiusAttr().Get()]))
                    elif prim.IsA(UsdGeom.Capsule):
                        capsule = UsdGeom.Capsule(prim)
                        radius, height = capsule.GetRadiusAttr().Get(), capsule.GetHeightAttr().Get()
                        axis = capsule.GetAxisAttr().Get()
                        align = {"Z": np.eye(3), "X": np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]]),
                                 "Y": np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]])}[axis]
                        rot = rot @ align
                        origin(element, position, rot)
                        ET.SubElement(geometry, "cylinder", radius=numbers([radius]), length=numbers([height]))
                        # URDF has no capsule primitive: the union of a cylinder and
                        # its two endpoint spheres has the same exterior surface.
                        for sign in (-1, 1):
                            cap = ET.SubElement(link, kind, name=element.attrib['name'] + f"_cap_{sign}")
                            origin(cap, position + rot[:, 2] * height / 2 * sign)
                            ET.SubElement(ET.SubElement(cap, "geometry"), "sphere", radius=numbers([radius]))
                            if kind == "visual":
                                ET.SubElement(ET.SubElement(cap, "material", name=cap.attrib['name'] + "_material"), "color", rgba=numbers(color(prim)))
                    else:
                        raise ValueError(f"Unsupported geometry: {prim.GetPath()}")
                if kind == "visual":
                    ET.SubElement(ET.SubElement(element, "material", name=element.attrib['name'] + "_material"), "color", rgba=numbers(color(prim)))
    limits = training_limits()
    for prim in stage.Traverse():
        if not prim.IsA(UsdPhysics.RevoluteJoint):
            continue
        joint = UsdPhysics.RevoluteJoint(prim)
        # This asset has identity child joint frames. Fail rather than silently
        # emit a wrong frame if a future USD changes that assumption.
        assert np.allclose(joint.GetLocalPos1Attr().Get(), 0)
        assert np.allclose(rotation(joint.GetLocalRot1Attr().Get()), np.eye(3))
        name = prim.GetName()
        element = ET.SubElement(robot, "joint", name=name, type="revolute")
        ET.SubElement(element, "parent", link=joint.GetBody0Rel().GetTargets()[0].name)
        ET.SubElement(element, "child", link=joint.GetBody1Rel().GetTargets()[0].name)
        origin(element, joint.GetLocalPos0Attr().Get(), rotation(joint.GetLocalRot0Attr().Get()))
        ET.SubElement(element, "axis", xyz=numbers({"X": [1, 0, 0], "Y": [0, 1, 0], "Z": [0, 0, 1]}[joint.GetAxisAttr().Get()]))
        values = {}
        for field, patterns in limits.items():
            matches = [v for pattern, v in patterns.items() if re.fullmatch(pattern, name)]
            assert len(matches) == 1, (name, field, matches)
            values[field] = matches[0]
        ET.SubElement(element, "limit", lower=numbers([math.radians(joint.GetLowerLimitAttr().Get())]),
                      upper=numbers([math.radians(joint.GetUpperLimitAttr().Get())]),
                      effort=numbers([values['effort_limit']]), velocity=numbers([values['velocity_limit']]))
        counts['joints'] += 1
    ET.indent(robot, space="  ")
    destination = OUTPUT / "waist_unlocked.urdf"
    ET.ElementTree(robot).write(destination, encoding="utf-8", xml_declaration=True)
    print(destination)
    print(counts)


if __name__ == "__main__":
    export()
