# !/usr/bin/env python
# -*- coding: utf-8 -*-

# Licensed to the Technische Universität Darmstadt under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The Technische Universität Darmstadt
# licenses this file to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import io
import os

from setuptools import setup, find_packages

# Package meta-data.
NAME = "inception-external-recommender"
DESCRIPTION = "INCEpTION external recommender library in Python"
HOMEPAGE = "https://inception-project.github.io/"
EMAIL = "inception-users@googlegroups.com"
AUTHOR = "The INCEpTION team"
REQUIRES_PYTHON = ">=3.8.0"

install_requires = [
    "flask",
    "filelock",
    "dkpro-cassis>=0.9.1",
    "joblib",
    "gunicorn",
    "deprecation",
]

contrib_dependencies = [
    "numpy>=1.24.2",
    "scikit-learn>=1.0.1",
    "sklearn_crfsuite~=0.3.6",
#    "rust_fst~=0.1.2",
    "spacy~=3.5.2",
    "nltk~=3.5",
    "jieba~=0.42",
    "sentence-transformers~=2.2.2",
    "lightgbm~=4.2.0",
    "diskcache~=5.2.1",
    "simalign~=0.4",
    "flair>=0.13.1"
]

test_dependencies = [
    "tox",
    "pytest",
    "codecov",
    "pytest-cov",
]

dev_dependencies = [
    "black",
    "wget"
]

doc_dependencies = [
    "sphinx",
    "sphinx-autodoc-typehints",
    "sphinx-rtd-theme"
]

extras = {
    "test": test_dependencies,
    "dev": dev_dependencies,
    "doc": doc_dependencies,
    "contrib": contrib_dependencies
}

here = os.path.abspath(os.path.dirname(__file__))

# Import the README and use it as the long-description.
# Note: this will only work if "README.rst" is present in your MANIFEST.in file!
try:
    with io.open(os.path.join(here, "README.md"), encoding="utf-8") as f:
        long_description = "\n" + f.read()
except FileNotFoundError:
    long_description = DESCRIPTION

# Load the package"s __version__.py module as a dictionary.
about = {}
with open(os.path.join(here, "ariadne", "__version__.py")) as f:
    exec(f.read(), about)


# Where the magic happens:
setup(
    name=NAME,
    version=about["__version__"],
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type="text/markdown",
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    url=HOMEPAGE,
    packages=find_packages(exclude="tests"),
    keywords="uima dkpro inception nlp",

    project_urls={
        "Bug Tracker": "https://github.com/inception-project/inception-external-recommender/issues",
        "Documentation": "https://github.com/inception-project/inception-external-recommender",
        "Source Code": "https://github.com/inception-project/inception-external-recommender",
    },

    install_requires=install_requires,
    test_suite="tests",

    tests_require=test_dependencies,
    extras_require=extras,

    include_package_data=True,
    license="Apache License 2.0",
    classifiers=[
        # Trove classifiers
        # Full list: https://pypi.python.org/pypi?%3Aaction=list_classifiers
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Topic :: Software Development :: Libraries",
        "Topic :: Scientific/Engineering :: Human Machine Interfaces",
        "Topic :: Text Processing :: Linguistic"
    ],

)