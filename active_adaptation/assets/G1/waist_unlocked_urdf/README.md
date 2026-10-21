# G1 training asset: URDF preview

Open `waist_unlocked.urdf` in VS Code and click the URDF Visualizer eye icon.
Keep the adjacent `meshes/` directory with the URDF; no ROS package lookup or
absolute mesh paths are required.

This file is exported directly from `../waist_unlocked.usd`, the Isaac asset
selected by `g1_waist_unlocked` in `facet/G1LocoFacet`. It is different from the
older `../g1_29dof_nohand/g1_29dof_nohand.urdf`.

The export contains 30 rigid-body links and 29 revolute joints. It preserves
link-frame visual geometry, basic diffuse colors, mass, centre of mass, full
inertia tensors, joint frames, axes and position limits. Mass is approximately
33.341142 kg. Effort and velocity limits come from the training configuration
in `active_adaptation/assets/humanoid.py`, which overrides some USD defaults.
Units are metres, kilograms, seconds and radians.

Collision geometry follows the USD: foot meshes are exported as convex hulls,
hand `boundingCube` approximations become boxes, and capsules become a cylinder
plus two endpoint spheres. Their exterior shape matches a capsule, but a physics
engine can generate different contact constraints for these separate primitives.
Other engines can also cook collision meshes differently from PhysX.

This is a geometry and rigid-body parameter export, not a guarantee of identical
simulation. URDF does not carry the full Isaac/PhysX solver configuration,
contact offsets, self-collision policy, PD gains, armature, action scaling,
delays, randomization, or trained policy. Material rendering is simplified from
MDL to diffuse RGBA. The viewer starts at zero joint angles; training uses the
initial pose specified in `humanoid.py` (including bent knees and elbows) and
places the pelvis at z = 0.85 m. Training continues to load the original USD.
The source USD's authored shoulder transforms differ slightly from its joint
zero frames (approximately 0.01 degrees). This export preserves the joint
frames rather than changing them to match the authored display pose.

Validation: ROS `check_urdf` parses the tree successfully. MuJoCo loads all
30 links and 29 joints; masses, centres of mass and full inertia tensors match
the source (maximum reconstructed inertia error below 4e-8 kg m²). Ten random
joint configurations match forward kinematics computed from the USD joint
frames (maximum position error below 1e-9 m and rotation-matrix error below
4e-9). All local mesh references resolve.

Regenerate from the repository root with a Python environment containing
`usd-core`, `numpy`, and `scipy`:

```sh
python scripts/export_g1_usd_urdf.py
```
