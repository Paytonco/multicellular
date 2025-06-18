from setuptools import find_packages, setup

setup(
    name="multicellular",
    version="0.1.0",
    packages=find_packages(),  # finds folders which have __init__.py
    install_requires=["numpy"],
    python_requires=">=3.11",
)
