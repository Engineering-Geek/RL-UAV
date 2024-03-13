UAV Multi-body Naming and Accessing
===================================

This document describes the naming conventions and access methods for the components of Unmanned Aerial Vehicles (UAVs) within a MuJoCo simulation environment. The UAV model consists of various bodies, motors, and sensors, each with a unique naming scheme to facilitate identification and manipulation during simulations.

Multi-body Components
---------------------

Bodies
^^^^^^

Each UAV is composed of a central body (chassis) and multiple propellers. The naming convention for the bodies is as follows:

- **Central Body**: The main body of the UAV is named as ``drone{index}``, where ``{index}`` is a unique identifier for each UAV in the simulation. For instance, the central body of the first UAV is named ``drone1``.

- **Propellers**: Each UAV has four propellers, named using the pattern ``drone{index}_prop{n}``, where ``{index}`` is the UAV identifier, and ``{n}`` is the propeller number (1 to 4). For example, the first propeller of the second UAV is named ``drone2_prop1``.

Motors
^^^^^^

Motors are attached to the propellers to control their rotation, providing thrust and maneuverability to the UAVs. The motors are named as ``drone{index}_motor{n}``, matching their respective propellers. For instance, the motor controlling the third propeller of the fourth UAV is named ``drone4_motor3``.

Sensors
^^^^^^^

Each UAV is equipped with sensors to measure acceleration, angular velocity, and orientation:

- **Accelerometer**: Named as ``drone{index}_imu_accel``, it measures the UAV's acceleration. For example, the accelerometer on the third UAV is named ``drone3_imu_accel``.

- **Gyroscope**: Named as ``drone{index}_imu_gyro``, it measures the UAV's angular velocity. The gyroscope on the first UAV is named ``drone1_imu_gyro``.

- **Orientation Sensor**: Named as ``drone{index}_imu_orientation``, it provides the UAV's orientation. For the second UAV, this sensor is named ``drone2_imu_orientation``.

Accessing UAV Components
------------------------

In a MuJoCo simulation, components of the UAVs can be accessed programmatically using their names. For instance, to apply a force to the first motor of the third UAV, you can use the following MuJoCo function:

.. code-block:: python

   mujoco.mj_applyFT(model, data, force, torque, point, object_name, frame)

where ``object_name`` would be ``"drone3_motor1"`` to target the specific motor.

Conclusion
----------

Understanding the naming conventions and access methods for UAV components in MuJoCo simulations is crucial for effective control and analysis. By adhering to a consistent naming scheme, developers can programmatically interact with each UAV and its components, enabling complex simulations and experiments in the field of robotics and AI.
