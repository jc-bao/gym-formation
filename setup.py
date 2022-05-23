from setuptools import setup, find_packages
from pathlib import Path

setup(
    name='formation_gym',
    author="Chaoyi Pan",
    author_email="pcy19@mails.tsinghua.edu.cn",
    version='0.0.1',
    description="An OpenAI Gym Env for Formaion",
    long_description=Path("README.md").read_text(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        ],
    packages=find_packages(
        exclude=["train"]
    ),
    python_requires='>=3.6'
)
