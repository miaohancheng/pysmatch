from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="pysmatch",
    use_scm_version=True,
    packages=find_packages(),
    description="Propensity Score Matching (PSM) on Python",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Miao HanCheng",
    author_email="hanchengmiao@gmail.com",
    url="https://github.com/mhcone/pysmatch",
    include_package_data=True,
    install_requires=[
        "matplotlib>=3.7.1",
        "numpy>=1.26.4,<2.0.0",
        "pandas>=2.1.4",
        "scipy>=1.13.1",
        "seaborn>=0.12.2",
        "statsmodels>=0.14.3",
        "scikit-learn>=1.5.2",
        "imbalanced-learn>=0.12.3",
    ],
    extras_require={
        "tree": ["catboost>=1.2.7"],
        "tune": ["optuna>=4.1.0"],
        "all": ["catboost>=1.2.7", "optuna>=4.1.0"],
    },
    python_requires=">=3.9",
)
