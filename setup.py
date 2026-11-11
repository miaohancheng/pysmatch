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
    url="https://github.com/miaohancheng/pysmatch",
    project_urls={
        "Documentation": "https://miaohancheng.com/pysmatch/",
        "Source": "https://github.com/miaohancheng/pysmatch",
        "Issues": "https://github.com/miaohancheng/pysmatch/issues",
        "Changelog": "https://github.com/miaohancheng/pysmatch/releases",
    },
    keywords=[
        "propensity score matching",
        "causal inference",
        "observational studies",
        "matching",
        "statistics",
    ],
    license="MIT",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    include_package_data=True,
    package_data={"pysmatch": ["py.typed"], "pysmatch.data": ["loan.csv"]},
    install_requires=[
        "numpy>=1.26.4",
        "pandas>=2.1.4",
        "scipy>=1.13.1",
        "scikit-learn>=1.5.2",
        "imbalanced-learn>=0.12.3",
    ],
    extras_require={
        "viz": [
            "matplotlib>=3.7.1",
            "seaborn>=0.12.2",
            "statsmodels>=0.14.3",
        ],
        "tree": ["catboost>=1.2.7"],
        "tune": ["optuna>=4.1.0"],
        "all": [
            "matplotlib>=3.7.1",
            "seaborn>=0.12.2",
            "statsmodels>=0.14.3",
            "catboost>=1.2.7",
            "optuna>=4.1.0",
        ],
        "dev": [
            "build",
            "pytest>=8",
            "pytest-cov>=5",
            "ruff>=0.8",
            "twine",
        ],
    },
    python_requires=">=3.9",
)
