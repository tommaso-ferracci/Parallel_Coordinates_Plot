import os
from setuptools import setup, find_packages

setup(
    name="par_coordinates",
    version="0.3",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.26.4",
        "matplotlib>=3.8.3",
        "pandas>=2.2.1",
    ],
    author="Tommaso Ferracci",
    url="https://github.com/tommaso-ferracci/Parallel_Coordinates_Plot",
    project_urls={
        "Documentation": "https://tommaso-ferracci.github.io/Parallel_Coordinates_Plot/index.html",
        "Source": "https://github.com/tommaso-ferracci/Parallel_Coordinates_Plots",
    },
    license="MIT",
    description="Create parallel coordinates plots of the hyperparameter search",
    long_description=open(
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "README.md"))
    ).read(),
    long_description_content_type="text/markdown",
    keywords=[
        "parallel coordinates",
        "parallel-coordinates",
        "machine learning",
        "deep learning",
        "hyperparameter",
        "visualization",
        "plot"
    ],
)