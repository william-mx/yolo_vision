from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='yolo_vision',
            executable='run_yolo',
            name='yolo_node',
            output='screen',
            parameters=[
                # Add parameters here if needed
            ],
            # arguments=['--ros-args', '--log-level', 'info'],  # Optional logging level
        )
    ])
