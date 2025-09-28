from setuptools import setup, find_packages
import pathlib

# Baca README.md untuk long description
here = pathlib.Path(__file__).parent.resolve()
long_description = (here / "README.md").read_text(encoding="utf-8")

setup(
    name="nr14-data-analyser",  # nama package yang di-install
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pandas>=1.3.0",
        "numpy>=1.21.0",
        "scikit-learn>=1.0.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0",
        "scikit-fuzzy>=0.4.2",
        "plotly>=5.0.0"
    ],
    author="Nur Rasyid",
    author_email="nurrasyid186@gmail.com",
    description="A comprehensive library for general data analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/nurrasyid14/General-Data-Analyser",
    project_urls={
        "Bug Tracker": "https://github.com/nurrasyid14/General-Data-Analyser/issues",
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
    python_requires=">=3.7",
)
