from setuptools import find_packages, setup

setup(
    name="quickseries",
    version="0.1.1",
    packages=find_packages(),
    url="https://github.com/millionconcepts/quickseries.git",
    author="Michael St. Clair",
    author_email="mstclair@millionconcepts.com",
    python_requires=">=3.11",
    install_requires=["dustgoggles", "numpy", "scipy", "sympy"],
    extras_require={"jit": "numba"}
)
