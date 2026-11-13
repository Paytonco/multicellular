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
        # Compiled, multi-threaded kernels for stepping cells, contact
        # mechanics and diffusion within a single run. 0.62+ for numpy 2.3.
        "numba>=0.62",
        # Keeps threaded BLAS from contending with numba's threads.
        "threadpoolctl",
        # Saving an animation as MP4: imageio provides the writer, and
        # imageio-ffmpeg bundles the ffmpeg binary it encodes with, so no
        # system ffmpeg install is needed.
        "imageio",
        "imageio-ffmpeg",
    ],
    python_requires=">=3.11",
)
