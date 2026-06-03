import os
from glob import glob
from setuptools import find_packages, setup

package_name = "sentinel_camera"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.launch.py")),
        (os.path.join("share", package_name, "config"), glob("config/*.yaml")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="SENTINEL",
    maintainer_email="kosukhin123@gmail.com",
    description="SENTINEL drone camera publishers (RGB now, NIR later).",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "rgb_camera = sentinel_camera.camera_publisher:main",
        ],
    },
)
