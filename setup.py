import pathlib as pl
from setuptools import setup, find_packages

HERE = pl.Path(__file__).parent

README_PATH = HERE / "README.md"

with open(README_PATH, "r") as rm:
    README_TEXT = rm.read()

setup(
    name="tapr",
    version="0.2.0",
    description="A library to facilitate tabular programming.",
    long_description=README_TEXT,
    long_description_content_type="text/markdown",
    url="https://bitbucket.org/elspacedoge/tapr",
    author="elspacedoge",
    author_email="elspacedoge@gmail.com",
    license="MIT",
    classifiers=[
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Operating System :: OS Independent"
    ],
    package_dir={"": "./"},
    packages=find_packages(where="./"),
    include_package_data=True,
    install_requires=["numpy", "pandas", "xarray", "h5py", "matplotlib", "plotly"]
)