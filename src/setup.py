from setuptools import setup, find_packages

setup(
    name="par_coordinates",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.26.4",
        "matplotlib>=3.8.3",
        "pandas>=2.2.1",
    ],
)