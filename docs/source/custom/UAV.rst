UAV Model Components in MuJoCo
==============================

The UAV model in MuJoCo is composed of several interconnected components, including the main body (chassis), propellers, motors, and sensors. This document provides detailed descriptions of each component's naming convention and its role in the UAV simulation.

UAV Components Overview
-----------------------

.. list-table::
   :widths: 20 80
   :header-rows: 1

   * - Component
     - Description
   * - ``drone``
     - The main body of the UAV, which acts as the chassis holding all other components.
   * - ``prop1``, ``prop2``, ``prop3``, ``prop4``
     - The propellers of the UAV, responsible for generating lift and controlling the UAV's movement.
   * - ``motor1``, ``motor2``, ``motor3``, ``motor4``
     - The motors controlling the rotation of each corresponding propeller.
   * - ``imu_accel``
     - The accelerometer sensor, providing acceleration data.
   * - ``imu_gyro``
     - The gyroscope sensor, offering angular velocity data.
   * - ``imu_orientation``
     - The orientation sensor, giving the UAV's current orientation.

Detailed Component Descriptions
-------------------------------

Bodies
^^^^^^

- **Main Body (`drone`)**: This body represents the UAV's chassis and serves as the central component to which all other parts are attached. It's defined with a mesh geometry (`drone.stl`), providing the UAV's physical appearance and collision properties.

Propellers
^^^^^^^^^^

- **Propellers (`prop1`, `prop2`, `prop3`, `prop4`)**: Each UAV has four propellers positioned at specific points relative to the main body. They are differentiated by their clockwise (CW) and counter-clockwise (CCW) rotation properties, affecting the UAV's dynamics. The propellers are essential for the UAV's lift and maneuverability.

Motors
^^^^^^

- **Motors (`motor1`, `motor2`, `motor3`, `motor4`)**: These actuators are attached to each propeller, controlling its rotation speed. The motor's `gear` attribute determines the relationship between the control signal and the propeller's rotation speed, impacting the thrust generated.

Sensors
^^^^^^^

- **Accelerometer (`imu_accel`)**: Mounted on the UAV's main body, this sensor measures linear acceleration in three dimensions, essential for understanding the UAV's movement and applying control algorithms.

- **Gyroscope (`imu_gyro`)**: This sensor provides the angular velocity, which is crucial for stabilizing the UAV and controlling its orientation.

- **Orientation Sensor (`imu_orientation`)**: It offers real-time data on the UAV's orientation, facilitating tasks that require knowledge of the UAV's current facing direction.

Cameras
^^^^^^^

- **Cameras (`camera_1`, `camera_2`)**: These cameras are positioned to capture the environment from the UAV's perspective. They are useful for tasks that require visual feedback, such as navigation and object detection.

Conclusion
----------

Understanding the structure and naming conventions of the UAV model in MuJoCo is essential for effectively controlling and experimenting with the simulation. Each component plays a specific role in the UAV's behavior, allowing for detailed customization and control of the UAV's actions in the virtual environment.
