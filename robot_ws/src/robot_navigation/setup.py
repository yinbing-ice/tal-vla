import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'robot_navigation'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
    ('share/ament_index/resource_index/packages',
        ['resource/' + package_name]),

    ('share/' + package_name, ['package.xml']),

    (os.path.join('share', package_name, 'launch'),
        glob('launch/*.py')),

    (os.path.join('share', package_name, 'config'),
        glob('config/*.yaml')),

    (os.path.join('share', package_name, 'maps'),
        glob('maps/*')),
        
    (os.path.join('share', package_name, 'urdf'),
    glob('urdf/*.urdf')),
],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='nvidia',
    maintainer_email='nvidia@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'navigate_to_pose_client = robot_navigation.navigate_to_pose_client:main',
            'generate_occupancy_map = robot_navigation.generate_occupancy_map:main',
        ],
    },
)
