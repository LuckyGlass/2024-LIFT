from setuptools import setup, find_packages

setup(
    name="my_local_library",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch==2.5.1",
        "transformers==4.47.1",
        "peft==0.14.0",
        "tqdm",
        "nltk"
    ]
)
