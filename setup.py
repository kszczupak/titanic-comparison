from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='luxoft-titanic',
    version='0.1.3',
    packages=find_packages(),
    python_requires='>=3.6',
    install_requires=[
       'pandas',
       'requests'
    ],
    url='',
    license='',
    author='Krzysztof Szczupak',
    author_email='szczupak.krzysztof@gmail.com',
    description='Titanic data comparision library',
    long_description=long_description,
    long_description_content_type="text/markdown",
    test_suite='tests'
)
