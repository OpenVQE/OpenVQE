import os, sys
from setuptools import setup, find_packages, find_namespace_packages
from setuptools.command.test import test as TestCommand

class PyTest(TestCommand):
    """
    A test command to run pytest on a the full repository.
    This means that any function name test_XXXX
    or any class named TestXXXX will be found and run.
    """
    def initialize_options(self):
        TestCommand.initialize_options(self)
        self.pytest_args = []

    def finalize_options(self):
        TestCommand.finalize_options(self)
        self.test_args = []
        self.test_suite = True

    def run_tests(self):
        import pytest
        errno = pytest.main([".", "-vv"])
        sys.exit(errno)

def get_description():
    """
    Returns the long description of the current
    package
    Returns:
        str
    """
    with open("README.md", "r") as readme:
        return readme.read()
    
    
setup(
    name="openvqe",
    version="0.0.1",
    author="Mohammad HAIDAR",
    license="GNU General Public License v3.0",
    description="OpenVQE package",
    long_description="test", #get_description(),
    long_description_content_type="text/markdown",
    url="https://github.com/Haidarmm/OpenVQE/tree/update-ferm",
    project_urls={
        "Bug Tracker": "https://github.com/Haidarmm/OpenVQE/issues",
        "Source code": "https://github.com/Haidarmm/OpenVQE/tree/update-ferm",
    },
    packages=find_namespace_packages(
        include=["openvqe.*", "openvqe.ucc_family.*", "openvqe.common_files.*", "openvqe.adapt.*", "openvqe.applications.*"]
    ),
    install_requires=["myqlm-fermion"],
    tests_require=["pytest"],
    cmdclass={'test': PyTest},
)
