from setuptools import setup, find_packages
import codecs
import os

parent_dir = os.path.abspath(os.path.dirname(__file__))

with codecs.open(os.path.join(parent_dir, "README.md"), encoding="utf-8") as readme:
    readme_description = "\n" + readme.read()


setup(
    name="imhblpce",
    version="1.0.0",
    author="mamdasn s",
    author_email="<mamdassn@gmail.com>",
    url="https://github.com/Mamdasn/imhblpce",
    description="This module attempts to enhance contrast of a given image by employing a method called HBLPCE.",
    long_description=readme_description,
    long_description_content_type="text/markdown",
    include_package_data=True,
    package_dir={"": "src"},
    py_modules=["imhblpce"],
    install_requires=[
        "numpy",
        "cvxpy",
        "numba",
        "imhist",
    ],
    keywords=[
        "python",
        "histogram",
        "imhblpce",
        "hblpce",
        "histogram equalization",
        "histogram based equalization",
        "histogram locality preserving",
        "contrast enhancement",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Programming Language :: Python :: 3.8",
        "Operating System :: OS Independent",
    ],
    scripts=["bin/imhblpce"],
)
