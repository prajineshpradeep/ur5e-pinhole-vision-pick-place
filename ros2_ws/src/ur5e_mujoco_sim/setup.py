import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'ur5e_mujoco_sim'

mjcf_files = [p for p in glob('mjcf/*') if os.path.isfile(p)]
mjcf_mesh_files = [p for p in glob('mjcf/meshes/*') if os.path.isfile(p)]

data_files = [
    ('share/ament_index/resource_index/packages',
        ['resource/' + package_name]),
    ('share/' + package_name, ['package.xml']),
    (os.path.join('share', package_name, 'mjcf'), mjcf_files),
]
if mjcf_mesh_files:
    data_files.append(
        (os.path.join('share', package_name, 'mjcf', 'meshes'), mjcf_mesh_files)
    )

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=data_files,
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='praji',
    maintainer_email='nith600e@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'mujoco_node = ur5e_mujoco_sim.mujoco_node:main',
            'vision_node = ur5e_mujoco_sim.vision_node:main',
        ],
    },
)
