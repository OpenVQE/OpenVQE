
## Introduction

OpenVQE is distributed under the MIT license. By contributing, you agree to license your work under MIT. 
To be part of our community, you can join our discord server([link here](https://discord.gg/fBr5MQ34)) or you can contact Dr. Mohammad( email: mohammadhaidar2016@outlook.com).

## Contribution checklist

Before adding your code to the OpenVQE here is a checklist that you must follow. If you have questions and/or you need help, you can contact nathan (nathanvaneberg@gmail.com) that will help you integrate your code.  

- [ ] The file `requirements.txt`([link here](https://github.com/OpenVQE/OpenVQE/blob/alpha/requirements.txt)) contains all package to needed and their version to run all the codes inside OpenVQE. You must be able to run all of your python files using this `requirements.txt`. If you want to change update the `requirements.txt` file, please indicate which package you want to add and which you want to change version. 
- [ ] Inside of you code there must be a notebook explaining how your package works and how to play with it.
- [ ] Apart if you asked to please put your code inside a nice folder in the application folder. For exemple if you want to code from a repo called `my_amazing_application` into OpenVQE, you must save your code inside the package at the path `openvqe/applications/my_amazing_application`.

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