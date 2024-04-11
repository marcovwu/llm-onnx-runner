import setuptools

from pip._internal.req import parse_requirements


# Readme
with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()


# Requirements
install_requires = [str(req.requirement) for req in parse_requirements('requirements.txt', session='hack')]


# Setup
setuptools.setup(
    name="llm-onnx-runner",
    version="0.1.0",
    author="MarcoWu",
    author_email="marcowu1999@gmail.com",
    description="export onnx and run",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/marcovwu/llm-onnx-runner",
    project_urls={
        "Source Code": "https://github.com/marcovwu/llm-onnx-runner",
        "Issue Tracker": "https://github.com/marcovwu/llm-onnx-runner/issues",
    },
    packages=setuptools.find_packages(),
    include_package_data=True,
    install_requires=install_requires,
    classifiers=[
        "Development Status :: 3 - Alpha",         # Develop Status
        "Intended Audience :: Developers",         # Target
        "Topic :: Multimedia :: Video",            # Items
        "License :: OSI Approved :: MIT License",  # Authentication
        "Programming Language :: Python :: 3",     # Code Language Version
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Operating System :: OS Independent",      # Support OS system
    ],
    python_requires='>=3.7'
)
print("Found packages:", setuptools.find_packages())
