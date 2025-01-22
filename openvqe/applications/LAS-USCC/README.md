# LAS-USCCSD:  A code to perform localized active space unitary selected coupled cluster calculations

## Overview
This repository contains the necessary code to run LAS-USCCSD (Localized Actice Space Unitary Selected Coupled Cluster with Single and Double excitations) calculations. It includes modified versions of the `mrh` and `QLAS-UCCSD` libraries tailored for specific computational tasks.

## Prerequisites
Before running the calculations, ensure you have the correct environment set up. This code is designed to run in a specific computational environment, which can be replicated using the `environment.yml` file provided in this repository.

## Setting Up Your Environment
To create an environment with the necessary dependencies, use the following command with Conda:

```bash
conda env create -f environment.yml

conda activate <env_name>

