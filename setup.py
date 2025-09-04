from setuptools import setup, find_packages

setup(
    name="MADD",
    version="0.1.0",
    packages=find_packages(include=["MADD", "MADD.*"]),
    install_requires=[
        "langchain",
        "langgraph"
        "openai",
    ],
    description="MADD package",
    classifiers=[
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.11",
)
