import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="table-evaluator",
    version="v1.4.2",
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
        'seaborn<=0.11.1',
        'matplotlib',
        'scikit-learn',
        'scipy'
    ],
    classifiers=[
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3 :: Only',
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
