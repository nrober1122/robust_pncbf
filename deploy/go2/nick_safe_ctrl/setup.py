from glob import glob
from pathlib import Path

from setuptools import setup

package_name = "nick_safe_ctrl"

setup(
    name=package_name,
    version="0.0.1",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages",
            ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", glob("launch/*.launch.py")),
        ("share/" + package_name + "/config", glob("config/*.yaml")),
        ("share/" + package_name + "/ckpts", glob("ckpts/*.pkl")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Yixuan",
    maintainer_email="yixuany@mit.edu",
    description="CBF obstacle-avoidance safety filter for Unitree Go2.",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "safe_ctrl_node = nick_safe_ctrl.safe_ctrl_node:main",
        ],
    },
)
