from setuptools import find_packages, setup

package_name = 'recorder'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Jusung Kim',
    maintainer_email='dsafjk0024@naver.com',
    description='Episode recorder: synchronized MCAP recording with keyboard start/stop/discard',
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'episode_recorder_node = recorder.episode_recorder_node:main',
            'keyboard_teleop = recorder.keyboard_teleop:main',
        ],
    },
)
