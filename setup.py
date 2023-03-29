from setuptools import setup, find_packages


test_deps = [
    "pytest>=6.2.3",
    "pytest-flask>=1.2.0",
    "pip>=21.0.1",
    "flake8>=3.9.2",
    "flake8-annotations>=2.6.2",
    "pytest-cov>=2.12.1",
    "black>=21.7b0",
    "markupsafe==2.0.1"
]

serve_deps = [
    "dploy-kickstart>=0.1.5",
]

extras = {"test": test_deps, "serve": serve_deps}

setup(
    name="mill-anomaly-detection",
    version="0.0.1",
    url="insus.ch",
    author="Gabriel Oliveira-Barra",
    author_email="gabriel@oliveira-barra.com",
    description="Take-Home test for anomaly detection in the Vertical Mill Project",
    packages=find_packages(),
    install_requires=["pandas>=1.3.2", "scikit-learn>=0.24.2", "matplotlib>=3.4.3", "seaborn>=0.11.2"],
    tests_require=test_deps,
    extras_require=extras,
)
