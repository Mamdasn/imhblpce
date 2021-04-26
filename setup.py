from setuptools import setup, find_packages
import codecs
import os

here = os.path.abspath(os.path.dirname(__file__))

with codecs.open(os.path.join(here, "README.md"), encoding="utf-8") as fh:
    long_description = "\n" + fh.read()


setup(
    name='imhblpce',
    version='0.0.3',
    author="mamdasn s",
    author_email="<mamdassn@gmail.com>",
    url="https://github.com/Mamdasn/imhblpce",
    description='This module attempts to enhance contrast of a given image by employing a method called HBLPCE.',
    long_description=long_description,
    long_description_content_type = "text/markdown",
    include_package_data=True,
    package_dir={'': 'src'},
    py_modules=["imhblpce"],
    install_requires=[
        "numpy", 
        "cvxpy",
        ],
    keywords=['python', 'histogram', 'imhblpce', 'hblpce', 'histogram equalization', 'histogram based equalization', 'histogram locality preserving', 'contrast enhancement'],
    classifiers=[
        "Development Status :: 1 - Planning",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
        "Operating System :: OS Independent",
    ]
)
