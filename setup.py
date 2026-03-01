import setuptools

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()


__version__ = '0.0.0'

REPO_NAME = "kidney-disease-classification-project"
AUTHOR_USER_NAME = "swatisingh0504005"
SRC_REPO = "kidney_disease_classification"
AUTHOR_EMAIL = "swatisingh0504005@gmail.com"

setuptools.setup(
    name=SRC_REPO,
    version=__version__,
    author=AUTHOR_USER_NAME,
    author_email=AUTHOR_EMAIL,
    description="A small python package for kidney disease classification",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url=f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}",
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src")
    

)