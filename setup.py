from setuptools import setup

dependencies = [
    'seaborn',
    'statsmodels',
    'scipy',
    'patsy',
    'matplotlib',
    'pandas',
    'numpy',
    'catboost' 
    ]

VERSION = "0.0.0.6"

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


