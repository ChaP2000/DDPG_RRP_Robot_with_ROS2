from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'rl_robot'

setup(
    name=package_name,
    version='0.0.0',
    packages=['rl_robot', 'reinforcement','reinforcement.utils'],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob(os.path.join('launch', '*launch.[pxy][yma]*'))),
        (os.path.join('share', package_name, 'urdf'), glob(os.path.join('urdf/*'))),
        (os.path.join('share', package_name, 'meshes'), glob(os.path.join('meshes/*'))),
        (os.path.join('share', package_name, 'config'), glob(os.path.join('config/*'))),
        (os.path.join('share', package_name, 'checkpoint'), glob(os.path.join('reinforcement/checkpoint/*'))),

    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='peerawit',
    maintainer_email='chawawiwat.p@gmail.com',
    description='Reinforcement learning of a robot manipulator with simulation using rviz2',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [

            'move_robot = rl_robot.move_robot:main',
            'train_model = reinforcement.train:main',
            'test_model = reinforcement.test:main'
        ],
    },
)
