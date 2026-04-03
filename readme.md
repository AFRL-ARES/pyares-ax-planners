# PyAres Ax Planners

This repository contains methods and examples for using Bayesian optimization planners based on Meta's Ax API with ARES OS using PyAres. It provides a compatibility layer to translate ARES OS planning requests into a format compatible with the Ax platform.

## Features
* **Custom Constraints**: Supports the use of implicit values within constraints and evaluates constraint strings via SymPy prior to planning.
* **Implicit Values & Parameters**: Supports planning with implicit variables in cases where constraints mandate that some pramters coming from ARES OS are dependent on other paramters. You can define implicit values using SymPy-compatible string expressions (e.g., `"flow_4 = total_flow - (flow_1 + flow_2 + flow_3)"`). 
* **Previous Data**: Can import prior experimental seed data from `.csv`, `.xls`, or `.xlsx` files. Column headers should match ARES OS provided parameter names, with the objective score value column labeled `objective`

## Currently Implemented Planners
* **Single Objective Bayesian Optimization (SOBO)**:`SOBO_Ax_Planner` provides single objective optimization for continuous variables.

* More to come!

## Installation & Environment Setup

This project requires **Python >=3.10**. A dedicated environment is highly recommended to keep module dependences for differnt PyAres services from causing issues.

### Option A: Using Anaconda / Miniconda / Miniforge (Recommended)

1.  **Create the environment:**
    ```bash
    conda create -n pyares_ax python=3.10 pip setuptools
    ```
2.  **Activate the environment:**
    ```bash
    conda activate pyares_ax
    ```
3.  **Install the package:**
    Navigate to the project root directory and run:
    ```bash
    pip install .
    ```
    *Note: This will automatically install dependencies listed in `pyproject.toml`.*

### Option B: Using Python venv

1.  **Ensure you have Python >=3.10 installed:**
    Check your version:
    ```bash
    python3 --version
    ```
2.  **Create the virtual environment:**
    ```bash
    python3 -m venv pyares_ax_venv
    ```
3.  **Activate the environment:**
    * **Windows:**
        ```powershell
        .\pyares_ax_venv\Scripts\activate
        ```
    * **Linux/macOS:**
        ```bash
        source pyares_ax_venv/bin/activate
        ```
4.  **Install the package:**
    ```bash
    pip install .
    ```


## Quick Start

To launch the planner service, run the provided startup script:
```bash
python start_pyares_ax_planner.py
```
This script initializes the `SOBO_Ax_Planner` and starts the `AresPlannerService` on port `1337`.

## Planner Settings

The `PyAres_Ax_Planner` exposes several settings that can be configured directly from ARES OS:

* **Minimize**: A boolean flag indicating whether the objective function should be minimized (Default: `False`).
* **RNG Seed**: A numerical setting to enforce a specific random seed for reproducibility.
* **Seed Data**: A string path pointing to a `.csv`, `.xls`, or `.xlsx` file containing historical trial data.
* **Constraints**: An array of string constraints evaluated by SymPy.
* **Implicit Values**: An array of string definitions for calculating implicit factors and parameters.
* **Parameter Value Type**: A string restricted to either `'Planned'` or `'Acheived'` (Default: `'Planned'`).
* **Verbose Output**: A boolean to toggle detailed logging in the console (Default: `True`).

## License
This project is licensed under the MIT License. Copyright (c) 2026 AFRL-ARES.