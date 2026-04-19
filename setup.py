"""Package setup for Intracranial Hemorrhage Detection."""

from setuptools import find_packages, setup

setup(
    name="ich-detection",
    version="0.1.0",
    description="Detection and classification of intracranial hemorrhage from CT scans",
    packages=find_packages(exclude=["tests*", "notebooks*", "scripts*"]),
    python_requires=">=3.10",
    install_requires=[],  # managed via requirements.txt
    extras_require={
        "dev": ["pytest", "ruff", "mypy", "pytest-cov"],
    },
    entry_points={
        "console_scripts": [
            "ich-prepare=scripts.prepare_data:main",
            "ich-train=scripts.train:main",
            "ich-evaluate=scripts.evaluate:main",
            "ich-predict=scripts.predict:main",
        ],
    },
)
