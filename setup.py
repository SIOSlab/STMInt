import setuptools
import os.path
import re

with open("README.md", "r") as fh:
    long_description = fh.read()

with open(os.path.join("STMint","__init__.py"), "r") as f:
    version_file = f.read()

version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]",\
        version_file, re.M)

if version_match:
    version_string = version_match.group(1)
else:
    raise RuntimeError("Unable to find version string.")

setuptools.setup(
    name="STMint",
    version=version_string,
    author="Jackson Kulik",
    author_email="jpk258@cornell.edu",
    description="State Transition Matrix Integrator",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/SIOSlab/STMInt",
    packages=setuptools.find_packages(exclude=['tests*']),
    include_package_data=True,
    install_requires=['numpy','scipy','astropy','sympy'],
    classifiers=[
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
    ]
)
