from setuptools import setup
import sys


setup(
    name='mnist',
    version="0.1",
    description="A package to download the MNIST digits dataset from Yan LeCunn's website",
    author="Patrick McCarthy",
    author_email="github@patrickmccarthy.cc",
    license='GPL-3.0',
    py_modules=['mnist',],
    install_requires=['numpy', 'requests',],
    zip_safe=True,
    long_description=open('README.md').read(),
)

