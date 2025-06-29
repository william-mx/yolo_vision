from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'yolo_vision'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='root',
    maintainer_email='william.engel@mdynamix.de',
    description='ROS2 node for real-time 2D object detection using Ultralytics YOLO.',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'run_yolo = yolo_vision.run_yolo:main'
        ],
    },
)
