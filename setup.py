from setuptools import setup

tests_requirements = ['pytest==7.1.2']
all_requirements = open("requirements.txt", "r").readlines() + tests_requirements


setup(
    name='Churner',
    version='0.1.0',
    author="Béranger GUEDOU",
    author_email="beranger@pillowanalytica.com",
    packages=[
        "churner",
        "churner.ml", 
        "churner.app", 
        "churner.tests"
        ],
    licence='LICENCE',
    description="Application that detects churner from input data.",
    long_description=open("README.md", "r").read(),
    install_required=all_requirements,
    classifiers=[
        "Development Status :: Beta",
        "Intended Audience :: Data scientists",
        "License :: OSI Approved :: Apache Software License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3.8"
    ]
)