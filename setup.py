import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="velour", 
    version="2020.11.18",
    author="Raphaël Tinarrage",
    author_email="raphael.tinarrage@gmail.com",
    description="Topological inference from point clouds with persistent homology",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/raphaeltinarrage/velour",
    packages=setuptools.find_packages(),
    install_requires=['gudhi', 'numpy', 'sklearn', 'scipy', 'matplotlib'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
