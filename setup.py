from setuptools import setup, find_packages
import os

def read_README():
    with open(os.path.join(os.path.dirname(__file__), 'README.md'), encoding='utf-8') as f:
        return f.read()
    
setup(
    name='diffindiff',
    version='2.4.0',
    description='diffindiff: Python library for convenient Difference-in-Differences analyses',
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    include_package_data=True,
    long_description=read_README(),
    long_description_content_type='text/markdown',
    author='Thomas Wieland',
    author_email='geowieland@googlemail.com',
    license_files=["LICENSE"],
    package_data={
        'diffindiff': ['tests/data/*'],
    },
    install_requires=[
        'numpy>=1.26.1,<=1.26.4',
        'pandas>=2.2.2,<3.0',
        'statsmodels==0.14.2',
        'scipy==1.15.3',
        'matplotlib',
        'scikit-learn>=1.3,<1.6',
        'xgboost>=1.7,<2.1',
        'lightgbm>=4.0,<4.5',
        'patsy>=0.5.6',
    ],
    test_suite='diffindiff.tests',
)