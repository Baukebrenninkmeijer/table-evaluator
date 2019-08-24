import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="TableEvaluator",
    version="0.0.1",
    author="Bauke Brenninkmeijer",
    author_email="bauke.brenninkmeijer@gmail.com",
    description="A package to evaluate synthetic datasets against real datasets.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/pypa/sampleproject",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)