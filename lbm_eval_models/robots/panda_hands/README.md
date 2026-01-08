A collection of hands to attach to Panda arms.

All of the hand models use the stock Panda hand (as shown here):

https://download.franka.de/documents/220010_Product%20Manual_Franka%20Hand_1.2_EN.pdf

They differ in various respects: type of finger, actuation, and geometry:
- `panda_hand_actuated.urdf`
  - Uses stock Panda fingers.
  - Collision geometry are a set of well-fit spheres.
  - The two fingers are independently actuated.
- `panda_hand_finray_actuated_mesh_mimic.urdf`
  - Uses TRI's custom implementation of fingers based on the Fin Ray Effect.
    (see e.g., https://advanced.onlinelibrary.wiley.com/doi/full/10.1002/aisy.202100133).
  - Collision geometry uses spheres on the hand, and tight meshes on the fingers.
  - The two fingers' actuation is coupled via a URDF mimic specification.
- `panda_hand_finray_actuated_spheres.urdf`
  - TRI's Fin Ray Effect fingers.
  - Collision geometry are a set of well-fit spheres on hand _and_ fingers.
  - The two fingers are independently actuated.

In addition to the model files themselves, there are some model directives
for attaching a hand to a panda arm (by welding it to the body  `panda_link8`).
They also add a diagnostic frame for defining a common calibration point.

- `add_panda_hand_finray_mimic.dmd.yaml`
- `add_panda_hand_finray_sphers.dmd.yaml`
- `add_panda_hand_nominal.dmd.yaml`

Applying these model directives requires that the directive
`panda_arms/add_panda_flange_rotated.dmd.yaml` has already been applied (the
hand is welded to the `panda::flange_rotated` frame).

Adds the calibration frame (applied by each of the other model directives).
- `add_panda_hand_calib_tip.dmd.yaml`
