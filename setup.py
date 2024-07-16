""" Setup of the quadruped-pend-gym package.
"""

from typing import List
import setuptools

_VERSION = "0.3.0"

# Short description.
short_description = "A gymnasium RL environment for balancing inverted pendulum on quadruped Unitree Go2"

# Packages needed for the environment to run.
# The compatible release operator (`~=`) is used to match any candidate version
# that is expected to be compatible with the specified version.
REQUIRED_PACKAGES = [
    "gymnasium == 1.0.0",
    "numpy ~= 1.24.4",
    "numpy-quaternion ~= 2023.0.3",
    "stable-baselines >= 2.4.0a5"
]

# Packages which are only needed for testing code.
TEST_PACKAGES = [

]  # type: List[str]

# Loading the "long description" from the projects README file.
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="quadruped-pend-gym",
    version=_VERSION,
    author="Pulak Gautam",
    author_email="pulakg21@iitk.ac.in",
    description=short_description,
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/pulak-gautam/quadruped-pend-gym",
    # Contained modules and scripts:
    packages=setuptools.find_packages(),
    package_data={"quadruped-pend-gym": ["models/go2/*"]}, #TODO: add other quadruped models
    install_requires=REQUIRED_PACKAGES,
    tests_require=REQUIRED_PACKAGES + TEST_PACKAGES,
    # PyPI package information:
    classifiers=[
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Operating System :: OS Independent",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development",
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    keywords=' '.join([
        "Quadruped-Pend"
        "Gymnasium",
        "Reinforcement-Learning",
        "Reinforcement-Learning-Environment",
    ]),
    entry_points={
        'console_scripts': [
            'quadruped_pend_gym = quadruped_pend_gym.stand_up:main',
        ],
    },
)
