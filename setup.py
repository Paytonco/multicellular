from setuptools import find_packages, setup

setup(
    name="multicellular",
    version="0.1.0",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        "numpy",
        "scipy",
        "pandas",
        "matplotlib",
        "tqdm",
        "pillow",
        "joblib",
        # Saving an animation as MP4: imageio provides the writer, and
        # imageio-ffmpeg bundles the ffmpeg binary it encodes with, so no
        # system ffmpeg install is needed.
        "imageio",
        "imageio-ffmpeg",
    ],
    python_requires=">=3.11",
)
