"""
 Copyright (c) 2026, StreamReady authors.
 SPDX-License-Identifier: CC-BY-NC-SA-4.0
 For full license text, see the LICENSE file in the repo root.
"""

import os

from setuptools import find_namespace_packages, setup


def fetch_long_description():
    here = os.path.abspath(os.path.dirname(__file__))
    readme = os.path.join(here, "README.md")
    if os.path.isfile(readme):
        with open(readme, encoding="utf-8") as f:
            return f.read()
    return ""


def fetch_requirements(filename):
    """Read a requirements file, dropping comments and blank lines."""
    here = os.path.abspath(os.path.dirname(__file__))
    with open(os.path.join(here, filename), encoding="utf-8") as f:
        lines = []
        for raw in f:
            line = raw.split("#", 1)[0].strip()
            if line:
                lines.append(line)
    return lines


setup(
    name="streamready",
    version="1.0.0",
    author="StreamReady authors",
    description=(
        "StreamReady: Task-Aware Hierarchical Q-Former for Enhanced Video "
        "Understanding (CVPR 2025)"
    ),
    long_description=fetch_long_description(),
    long_description_content_type="text/markdown",
    keywords=[
        "video understanding",
        "multimodal large language model",
        "question answering",
        "video captioning",
        "q-former",
    ],
    # StreamReady's own code is CC BY-NC-SA 4.0 (non-commercial research use).
    # The vendored `lavis/` library stays BSD 3-Clause. LICENSE Parts A and B
    # list exactly which files fall on each side, and Part C is the patent notice.
    license="CC-BY-NC-SA-4.0",
    # The `lavis` package is the vendored Salesforce LAVIS library that
    # StreamReady is built on; it ships inside this repo rather
    # than being installed from PyPI, because the model, dataset and config
    # code live inside it.
    packages=find_namespace_packages(include=["lavis", "lavis.*"]),
    include_package_data=True,
    install_requires=fetch_requirements("requirements.txt"),
    python_requires=">=3.8",
    classifiers=[
        "Intended Audience :: Science/Research",
        "License :: Other/Proprietary License",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    zip_safe=False,
)
