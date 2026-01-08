The Cabot station consists of two Panda arms welded to a table (in addition to
a collection of assorted cameras).

There are two flavors: "nominal" and "simulation". They differ in the nature of
the grippers that get affixed to the Panda arms. Both use Schunk grippers with
TRI's bespoke Fin Ray fingers. The _simulation_ flavor includes flir cameras
an their attendant mounting plates, moving the gripper farther away from
the arm's distal link.

Beyond that, there are several variants of the simulation flavor to support
different appearances of the table.

 - `add_cabot_nominal.dmd.yaml` - the "nominal" variant as described above.
 - `add_cabot_simulation_dmd.yaml` - the "simulation" variant as described above.
  - `add_cabot_simulation_cherry_red_table.dmd.yaml`
    - Table top is "cherry red" instead of "natural brown".
  - `add_cabot_simulation_dark_brown_table.dmd.yaml`
    - Table top is "dark brown" instead of "natural brown"

The remaining model directive files support the configurations above:
  - `add_cabot_with_grippers.dmd.yaml`
    - Builds robot on table with Flir cameras and Schunk grippers.
    - Requires that the table has already been added.
  - `add_cabot_without_grippers.dmd.yaml`
    - Builds robot on table without any grippers.
    - Requires that the table has already been added.
  - `add_cherry_red_table_top.dmd.yaml`
    - Specifies the "cherry red" table and the robot weld points.
  - `add_dark_brown_table_top.dmd.yaml`
    - Specifies the "dark brown" table and the robot weld points.
  - `add_natural_brown_table_top.dmd.yaml`
    - Specifies the "natural brown" (default) table and the robot weld points.
  - `add_table_top_weld.dmd.yaml`
    - Defines the weld points on the table.
