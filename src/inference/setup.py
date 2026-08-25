from setuptools import find_packages, setup

package_name = 'inference'

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
    description='Sync ACT inference node: observation in, joint-command action out',
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'sync_inference_node = inference.sync_inference_node:main',
        ],
    },
)
