
## Introduction

OpenVQE is open-source software licensed under the MIT license. By contributing, you agree to license your work under MIT. To join our community, you can contact Dr. Mohammad Haidar directly at [mohammadhaidar2016@outlook.com](mailto:mohammadhaidar2016@outlook.com) or nathan at [nathanvaneberg@gmail.com](mailto:nathanvaneberg@gmail.com).

## Code Integration Checklist

Before contributing, we recommend installing the OpenVQE package (see instructions below), exploring the example notebooks, reviewing the documentation and understanding the dependencies required to run the program.

Once you're familiar with OpenVQE, follow this checklist to make integrating your code as smooth as possible. For assistance, feel free to reach out to Nathan Vaneberg at [nathanvaneberg@gmail.com](mailto:nathanvaneberg@gmail.com).

- [ ] **Requirements File**: The `requirements.txt` file([link here](https://github.com/OpenVQE/OpenVQE/blob/alpha/requirements.txt)) includes all package to needed and their version to run all the codes inside OpenVQE. You should be able to run all the code you want to integrate using this `requirements.txt`. If you need to update `requirements.txt`, please specify which packages you want to add or which versions you want to modify. 

- [ ] **Documentation Notebook**: Include a Jupyter notebook to your application that explains how your code works and provides examples of how to use it.

- [ ] **Code Placement**: Unless otherwise specified, place your code in a designated folder within the `applications` directory. For example, if you are adding code from a repository called `my_amazing_application`, save it under `openvqe/applications/my_amazing_application`.
    
- [ ] **Agree to MIT License**: Send an email to Mohammad Haidar (<mohammadhaidar2016@outlook.com>), with Nathan Vaneberg (<nathanvaneberg@gmail.com>) cc'd. Include a link to your pull request, a description of your changes, and a statement confirming that your contribution will be licensed under MIT.

- [ ] **Add a description of your work inside the documentation**: You can add a description of your work and an introduction of yourself in our documentation (coming soon). Please contact <huybinhfr4120@gmail.com> for instructions on how to contrbute to the documentation.

## How to contribute:

- Go to the OpenVQE main page and click the fork button.

![alt text](images/image-6.png)
- Deselect "Copy the main branch only".
- Click on "Choose an owner" and select a github profile.
- Click on create fork.
- Open a terminal.
- Install your fork: 
```shell
git clone https://github.com/[your gitlab username]/OpenVQE.git
cd OpenVQE
git checkout alpha
pip install .
pip install -r requirements.txt
```
- Add your new amazing functionnalities and push your changes: 
```shell
git push origin HEAD
```
- Open a pull request (PR) to the alpha branch of the OpenVQE repository.
    - Go to your github fork.
    - Click on the contribute button.

    ![alt text](images/image.png)
    - Click on open pull request.

    ![alt text](images/image-2.png)
    - Open a pull request (PR) from your forked repository to the alpha branch of the OpenVQE repository.

    ![alt text](images/image-3.png)
    - Click on create pull request.

    ![alt text](images/image-4.png)
- Finally, send an email to Mohammad Haidar (mohammadhaidar2016@outlook.com) with Nathan Vaneberg (nathanvaneberg@gmail.com) cc'd. Include a link to your PR, a description of your changes, and confirm your contribution will be licensed under MIT.
