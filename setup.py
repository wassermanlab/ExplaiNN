"""
ExplaiNN
interpretable and transparent neural networks for genomics
"""

# Always prefer setuptools over distutils
from setuptools import setup, find_packages

# To use a consistent encoding
from codecs import open
from os import path

# The directory containing this file
# HERE = path.abspath(path.dirname(__file__))
#
# # Get the long description from the README file
# with open(path.join(HERE, 'README.md'), encoding='utf-8') as f:
#     long_description = f.read()

long_description = "interpretable and transparent neural networks for genomics"

setup(
    name="explainn",
    version="0.1.5",
    description=long_description,
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Gherman Novakovsky @ wassermanlab",
    author_email="gnovakovsky@cmmt.ubc.ca",
    license="MIT",
    classifiers=[
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Operating System :: OS Independent"
    ],
    packages=find_packages(), # ["explainn"],
    include_package_data=True,
    install_requires=["numpy", "h5py", "tqdm", "pandas"]
)