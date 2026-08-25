from setuptools import find_packages, setup

package_name = 'converter'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test', 'tests']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/config', ['config/so101.yaml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Jusung Kim',
    maintainer_email='dsafjk0024@naver.com',
    description='Convert recorded rosbag episodes into LeRobot v3.0 datasets',
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'rosbag-to-lerobot = converter.cli:main',
        ],
    },
)
