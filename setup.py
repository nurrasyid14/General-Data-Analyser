from setuptools import setup, find_packages

setup(
    name="govdata_analyser",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pandas>=1.3.0",
        "numpy>=1.21.0",
        "scikit-learn>=1.0.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0",
    ],
    author="Nur Rasyid",
    author_email="nurrasyid186@gmail.com",
    description="A comprehensive library for general data analysis",
    python_requires=">=3.7",
)