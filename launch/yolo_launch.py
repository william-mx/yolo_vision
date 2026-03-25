from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    # ── Declare launch arguments ──────────────────────────
    publish_plot_arg = DeclareLaunchArgument(
        'publish_plot',
        default_value='true',
        description='Publish annotated debug plot image on /result'
    )
    publish_mask_arg = DeclareLaunchArgument(
        'publish_mask',
        default_value='false',
        description='Publish segmentation mask on /mask and label mapping on /label_mapping'
    )
    publish_detections_arg = DeclareLaunchArgument(
        'publish_detections',
        default_value='false',
        description='Publish Detection2DArray on /detections_2d and label mapping on /label_mapping'
    )

    return LaunchDescription([
        publish_plot_arg,
        publish_mask_arg,
        publish_detections_arg,
        Node(
            package='yolo_vision',
            executable='run_yolo',
            name='yolo_node',
            output='screen',
            parameters=[{
                'publish_plot':       LaunchConfiguration('publish_plot'),
                'publish_mask':       LaunchConfiguration('publish_mask'),
                'publish_detections': LaunchConfiguration('publish_detections'),
            }],
        )
    ])