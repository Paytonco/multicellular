from setuptools import find_packages, setup

setup(
    name="multicellular",
    version="0.1.0",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=["numpy", "pandas", "matplotlib", "tqdm", "pillow"],
    python_requires=">=3.11",
)
