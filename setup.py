import setuptools


__version__ = "0.0.0"

REPO_NAME = "Kidney_Disease_Classification"
Author_USER_NAME = "Group16"
SRC_REPO = "cnnClassifier"

setuptools.setup(
    name=f"{REPO_NAME}-{Author_USER_NAME}-{SRC_REPO}",
    version=__version__,
    author=Author_USER_NAME,
    author_email="",
    description="A small package for Kidney Disease Classification",
    long_description="",
    long_description_content_type="text/markdown",
    url=f"",
    project_urls={
        "Bug Tracker": f"",
    },
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
)
