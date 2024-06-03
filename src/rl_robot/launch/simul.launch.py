from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, ExecuteProcess, RegisterEventHandler, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os 
import xacro
from launch_ros.parameter_descriptions import ParameterValue
from launch.substitutions import Command
from launch.event_handlers import OnExecutionComplete, OnProcessStart

def generate_launch_description():

    urdf_path = os.path.join(get_package_share_directory("rl_robot"),
                             'urdf','rrp.urdf.xacro')
    
    robot_desc = xacro.process_file(urdf_path).toxml()

    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name = "robot_state_publisher",
        output = "screen",
        parameters= [{'robot_description': robot_desc, 'use_sim_time' : False # True if using gazebo
                      }]
    )
    #run executable

    #Launch Rviz2 with executable using OnProcessStart

    rviz2 = ExecuteProcess(
        cmd = ["rviz2 rviz2 -d", os.path.join(get_package_share_directory("rl_robot"), "config", "rl_robot_config.rviz")],
        shell = True,
    )

    move_robot = Node(
        package = "rl_robot",
        executable = "move_robot",
        name = 'move_robot',
    )


    return LaunchDescription([

        robot_state_publisher,

        rviz2,

        move_robot,

    ])