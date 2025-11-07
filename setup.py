from setuptools import setup, find_packages


setup(
    name="grasp",
    version="0.1.0",
    description="Optimize ICL examples using evolutionary algorithms",
    author="..",
    author_email="your.email@example.com",
    packages=find_packages(),
    install_requires=[
        line.strip()
        for line in open("requirements.txt")
        if line.strip() and not line.startswith("#")
    ],
    python_requires=">=3.7",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)
