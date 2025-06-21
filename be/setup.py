from setuptools import setup, find_packages

setup(
    name="spectra",
    version="0.1.0",
    author="Ryan, Mizan, Hunter", # Consolidated from pyproject.toml
    author_email="mizantompkins@gmail.com", # Using one email, can be a generic project email
    description="A library for logging and analyzing LangGraph workflows.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="YOUR_PROJECT_URL_HERE",  # Replace with your project's URL if you have one
    packages=find_packages(include=['spectra', 'spectra.*']),
    install_requires=[
        "langgraph>=0.4.5,<0.5.0",
        "langchain-openai>=0.3.17,<0.4.0",
        "langchain>=0.3.25,<0.4.0",
        "python-dotenv>=1.1.0,<2.0.0"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License", # Assuming MIT, change if different
        "Operating System :: OS Independent",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.12,<4.0",
) 