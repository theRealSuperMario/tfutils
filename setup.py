from setuptools import setup, find_packages

setup(
    name="tfutil",
    version="0.1",
    description="collection of tensorflow snippets",
    url="https://github.com/theRealSuperMario/tfutils",
    author="Sandro Braun",
    author_email="supermario94123@gmail.com",
    license="MIT",
    packages=find_packages(),
    install_requires=["matplotlib", "numpy", "scikit-image", "scipy", "tensorflow"],
    zip_safe=False,
)
