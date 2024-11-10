from setuptools import setup
import os

dependencies = [
    'catboost>=1.2.7',
    'matplotlib>=3.7.1',
    'numpy>=1.26.4,<2.0.0',
    'pandas>=2.1.4',
    'patsy>=0.5.6',
    'scipy>=1.13.1',
    'seaborn>=0.12.2',
    'statsmodels>=0.14.3',
    'scikit-learn>=1.5.2',
    'imbalanced-learn>=0.12.3'
    ]

with open('README.md', 'r') as f:
    long_description = f.read()

setup(
    name='pysmatch',
    packages=['pysmatch'],
    version=os.getenv('PACKAGE_VERSION', '0.1'),
    description='Propensity Score Matching(PSM) on python',
    author='Miao HanCheng',
    author_email='hanchengmiao@gmail.com',
    long_description_content_type='text/markdown',
    long_description=long_description,
    url='https://github.com/mhcone/pysmatch',
    download_url=f'https://github.com/mhcone/pysmatch/archive/v{os.getenv("PACKAGE_VERSION", "0.1")}.tar.gz',
    keywords=['logistic', 'regression', 'matching', 'observational', 'machine learning', 'inference','pysmatch','Propensity Score Matching'],
    include_package_data=True,
    install_requires=dependencies
)


