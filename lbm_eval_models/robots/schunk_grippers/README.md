A collection of attachments (cameras and grippers) and some model directives
for applying them to Panda arms.

This includes Flir cameras
(https://www.teledynevisionsolutions.com/products/blackfly-s-gige/?segment=iis&vertical=machine%20vision), Schunk grippers
(https://schunk.com/de/en/gripping-systems/parallel-gripper/wsg/c/PGR_820), and
TRI's bespoke Fin Ray Effect fingers.

There are two different models of the flir cameras. Both versions include two
cameras fixed to a mounting plate which attaches to the panda arm. They differ
in the specification of the collision geometry:
 - `flir_assembly_primitives.sdf`
    - Primitive collision geometry around camera bodies and plate (other parts
      are not represented).
 - `flir_assembly_spheres.sdf`
    - Sphere collision geometry around cameras (but not plate).

The grippers both include TRI's bespoke Fin Ray fingers. They differ in
actuation and collision geometry:
 - `schunk_wsg_50_finray_actuated_mimic_simulation.sdf`
    - Collision geometry are meshes.
    - A single actuation DoF drives both finger positions.
 - `schunk_wsg_50_finray_actuated_planning.urdf`
    - Collision geometry fitted spheres.
    - Each finger is individually actuated.
 Note: the two `schunk_wsg_50_finray_actuated_*` models are _not_ oriented the
 same. They differ by a 90-degree rotation around Wx.

 - `panda_schunk_wsg_flange.sdf`
    - The mass and visual properties of a flange.
    - No collision geometry

In addition to the model files themselves, there are some model directives
for attaching a hand to a panda arm. They all weld the gripper to the frame
`panda::flange_rotated` which can be introduced by the directive file
`panda_arms/add_panda_flange_rotated.dmd.yaml`.

 - `add_flir_schunk_wsg_mimic_simulation.dmd.yaml`
   - Adds the schunk hand (`schunk_wsg_50_finray_actuated_mimic_simulation.sdf`)
   - Adds the flir camera (`flir_assembly_primitives.sdf`)
   - All attached bodies are placed into a collision filter group (and avoids
     collision with `panda_wrist_filter_group` -- i.e., end of arm).
 - `add_panda_hand_schunk_wsg.dmd.yaml`
   - Adds attachment flange (`panda_schunk_wsg_flange.sdf`)
   - Adds schunk hand (`schunk_wsg_50_finray_actuated_planning.urdf`)
   - Adds calibration frame (`add_schunk_calib_tip.dmd.yaml`)
 - `add_schunk-calib_tip.dmd.yaml`
   - Adds calibration frame.
