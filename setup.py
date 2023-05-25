"""
ExplaiNN: interpretable and transparent neural networks for genomics
"""

import setuptools

with open("README.md", "r") as f:
    long_description = f.read()

setuptools.setup(
    name="explainn",
    version="23.5.1",
    author="Gherman Novakovsky, Oriol Fornes",
    author_email="g.e.novakovsky@gmail.com, oriol.fornes@gmail.com",
    description="ExplaiNN: interpretable and transparent neural networks for genomics",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/wassermanlab/ExplaiNN",
    packages=setuptools.find_packages(),
    include_package_data=True,
    classifiers=[
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=["numpy", "h5py", "tqdm", "pandas"]
)