import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="ifeel", # Replace with your own username
    version="1.5.1",
    author="Maomao Hu",
    author_email="maomao.hu@eng.ox.ac.uk",
    description="A python package for Interpretable Feature Extraction of Electricity Loads (IFEEL)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/chacehoo/IFEEL",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)
