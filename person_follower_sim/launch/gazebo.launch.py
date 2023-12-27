#! /usr/bin/env python3
import xacro
from os.path import join
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription, TimerAction, LogInfo
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory
from launch.substitutions import LaunchConfiguration, Command, PythonExpression
import launch.conditions
from launch_ros.descriptions import ParameterValue

def generate_launch_description():
    
    #path to xacro file
    xacro_file=get_package_share_directory('person_follower_sim')+'/urdf/mr_robot.xacro'
    my_pkg_dir = join(get_package_share_directory('person_follower_sim'), 'worlds', 'model.sdf')
    actor_world = join(get_package_share_directory("person_follower_sim"),'worlds','stand.world')
    actor_world1 = join(get_package_share_directory("person_follower_sim"),'worlds','walk.world')

    # Include the gazebo.launch.py file
    gazebo=IncludeLaunchDescription(
        PythonLaunchDescriptionSource([get_package_share_directory('gazebo_ros'), '/launch/gazebo.launch.py']),
        launch_arguments={'pause': 'true',
                          'world':actor_world}.items()
    )

    #spawn world
    world=Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        name='spawn_model',
        output='screen',
        arguments=['-entity', 'first_world', '-file', my_pkg_dir],
    )
    #publishing robot_state into topic robot_description
    robot_state=Node(package = 'robot_state_publisher',
                            executable = 'robot_state_publisher',
                            name='robot_state_publisher',
                            parameters = [{'robot_description': ParameterValue(Command( \
                                        ['xacro ', xacro_file,
                                        # ' kinect_enabled:=', kinect_enabled,
                                        # ' lidar_enabled:=', lidar_enabled,
                                        # ' camera_enabled:=', camera_enabled,
                                        ]), value_type=str)}]
                            )
    #swawn mr_robot using the topic "/mr_robot_description"
    robot_spawn=Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        name='spawn_mr_robot',
        output='screen',
        arguments=[
    '-entity', 'mr_robot',
    '-topic', '/robot_description',
    '-x', '8.0',  # Set the initial X position
    '-y', '-8.0',  # Set the initial Y position
    '-z', '0.0' ,  # Set the initial Z position
    '-Y', '-3.14'   # Set the initial Z position
]
    )
    return LaunchDescription([
        gazebo,
        world,
        robot_state,
        robot_spawn
    ])













