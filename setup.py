import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="table-evaluator",
    version="v1.3",
    author="Bauke Brenninkmeijer",
    author_email="bauke.brenninkmeijer@gmail.com",
    description="A package to evaluate how close a synthetic data set is to real data.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Baukebrenninkmeijer/Table-Evaluator",
    packages=setuptools.find_packages(),
    install_requires=[
        'pandas',
        'numpy',
        'tqdm',
        'psutil',
        'dython==0.5.1',
        'seaborn',
        'matplotlib',
        'scikit-learn',
        'scipy'
    ],
    classifiers=[
        "Programming Language :: Python :: 3.7",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
