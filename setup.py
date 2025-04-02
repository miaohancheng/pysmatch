from setuptools import setup

with open("README.md", "r") as f:
    long_description = f.read()

setup(
    name="pysmatch",
    use_scm_version=True,
    setup_requires=["setuptools_scm"],
    packages=["pysmatch"],
    description="Propensity Score Matching (PSM) on Python",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Miao HanCheng",
    author_email="hanchengmiao@gmail.com",
    url="https://github.com/mhcone/pysmatch",
    include_package_data=True,
    install_requires=[
        "catboost>=1.2.7",
        "matplotlib>=3.7.1",
        "numpy>=1.26.4,<2.0.0",
        "pandas>=2.1.4",
        "scipy>=1.13.1",
        "seaborn>=0.12.2",
        "statsmodels>=0.14.3",
        "scikit-learn>=1.5.2",
        "imbalanced-learn>=0.12.3",
        "optuna>=4.1.0"
    ],
)