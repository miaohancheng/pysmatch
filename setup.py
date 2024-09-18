from setuptools import setup

dependencies = [
    'catboost>=1.2.7',
    'matplotlib>=3.7.1',
    'numpy>=1.26.4',
    'pandas>=2.1.4',
    'patsy>=0.5.6',
    'scipy>=1.13.1',
    'seaborn>=0.12.2',
    'statsmodels>=0.14.3'
    ]

VERSION = "0.1"

setup(
    name='pysmatch',
    packages=['pysmatch'],
    version=VERSION,
    description='Matching techniques for Observational Studies',
    author='Miao HanCheng',
    author_email='hanchengmiao@gmail.com',
    url='https://github.com/mhcone/pysmatch',
    download_url='https://github.com/mhcone/pysmatch/archive/{}.tar.gz'.format(VERSION),
    keywords=['logistic', 'regression', 'matching', 'observational', 'study', 'causal', 'inference','pysmatch'],
    include_package_data=True,
    install_requires=dependencies
)


