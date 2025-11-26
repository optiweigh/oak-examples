# ROS DRIVER BASIC EXAMPLE

This is an example of ROS driver being run as an OAK 4 app. It launches ROS driver in standalone and publishes RGB Pointcloud.
These topics can be accessible for viewing and/or further processing directly on host.
It is based on ROS2 Kilted, but you should be able to subscribe to topics in other distributions such as Humble or Jazzy.

## Prerequisites

Before you begin, ensure you have the following installed on your host machine:

- ROS2 (Humble, Jazzy, or Kilted)
- Rviz2

## Setup Instructions

1. Install the required ROS packages:

   ```bash
   sudo apt update
   sudo apt install ros-<distro>-rviz2
   ```

2. Source your ROS2 environment:

   ```bash
   source /opt/ros/<distro>/setup.bash
   ```

## Running the Example

1. Launch the OAK 4 app with the ROS driver:

   ```bash
   cd <path_to_this_example>
   oakctl app run .
   ```

2. In a new terminal, source your ROS2 environment and run Rviz2:

   ```bash
   source /opt/ros/<distro>/setup.bash
   rviz2
   ```

## Visualizing Data in Rviz

1. In Rviz, add a new display by clicking the "Add" button in the bottom toolbar.
   Note, you can add by display type, or just by topic which will automatically add the correct display type.

2. For RGB Pointcloud visualization:

   - Add a "Pointcloud" display
   - Set the topic to `/oak/rgbd/points`
   - Adjust the display settings as needed

3. Save your Rviz configuration for future use by clicking "File" > "Save Config".

## Troubleshooting

If you encounter issues with topic names or data types, verify that your ROS2 distribution matches the one used in the example. You may need to adjust topic names or data types accordingly.
