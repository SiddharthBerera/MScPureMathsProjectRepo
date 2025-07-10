from setuptools import setup, find_packages

setup(
    name = "allen_cahn_pde_solver",
    version = "0.1",
    packages = find_packages(),
    install_requires = ["torch", "scipy", "meshzoo"]
)