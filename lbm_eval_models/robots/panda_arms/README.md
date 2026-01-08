A collection of model directives for working with Panda arms.

These model directives require that a Panda arm has already been added to the
model. This can be achieved by adding models such as:

  - package://drake_models/franka_description/urdf/panda_arm_link5split.urdf
  - package://drake_models/franka_description/urdf/panda_arm.urdf

They have the following effect:

 - `add_panda_flange_rotated.dmd.yaml`
   - Adds helpful frames to panda arm.
 - `add_panda_arm_extras.dmd.yaml`
   - Augments panda arm with useful frames and collision filter groups.
