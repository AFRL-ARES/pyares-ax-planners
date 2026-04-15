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

This project requires **Python >=3.11**. A dedicated environment is highly recommended to keep module dependences for differnt PyAres services from causing issues.

### Option A: Using Anaconda / Miniconda / Miniforge (Recommended)

1.  **Create the environment:**
    ```bash
    conda create -n pyares_ax python=3.11 pip setuptools
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

The `PyAres_Ax_Planner` exposes several settings by defualt that can be configured directly from ARES OS:

* **Seed Data**: A string path pointing to a `.csv`, `.xls`, or `.xlsx` file containing historical trial data.
* **Constraints**: An array of string constraints evaluated by SymPy.
* **Implicit Values**: An array of string definitions for calculating implicit factors and parameters.
* **Parameter Value Type**: A string restricted to either `'Planned'` or `'Acheived'` (Default: `'Planned'`).
* **Verbose Output**: A boolean to toggle detailed logging in the console (Default: `True`).

Specific planner instances can implement additonal settings.

Here is the section you can add to the README file to guide developers on how to implement their own custom planners using the framework.

***

## Implementing a Custom Planner

You can easily create your own Ax-based planner for ARES OS by subclassing the `PyAres_Ax_Planner` base class. This base class handles all the heavy lifting of translating PyAres requests, processing historical/seed data into pandas DataFrames, and managing implicit constraints. 

To implement a custom planner, you need to follow these steps:

### 1. Subclass `PyAres_Ax_Planner` and Override Initialization Attributes
In your subclass's `__init__` method, call `super().__init__()` and then override the placeholder attributes with your planner's specific information.

```python
from pyares_Ax.ax import PyAres_Ax_Planner
from PyAres import AresDataType

class MyCustom_Ax_Planner(PyAres_Ax_Planner):
    def __init__(self):
        super().__init__()
        # Override generic planner info
        self.name = "My Custom Ax Planner"
        self.description = "A tailored Ax Bayesian Optimization planner."
        self.version_number = "1.0.0"
        
        # Link to your custom planning function
        self.plan_function = my_custom_planner_logic 
        
        # Add any planner-specific settings exposed to ARES OS
        self.add_setting('Minimize', AresDataType.BOOLEAN, False)
        self.add_setting('Custom Iterations', AresDataType.NUMBER, default_value=10)
```

### 2. Override `_configure_objectives()`
The base class contains a `_configure_objectives` method that acts as a placeholder. You are expected to override this method in your child class to properly configure the Ax `ObjectiveProperties` based on the settings provided by ARES OS.

```python
    def _configure_objectives(self):
        # Example: Setting objective direction based on a user setting
        minimize_flag = self.settings.get('Minimize', False)
        self.objectives = {'objective': ObjectiveProperties(minimize=minimize_flag)}
```

### 3. Define Your `plan_function`
The `plan_function` is the core routine of your planner. It is called by the compatibility layer and is provided with fully parsed and formatted Ax API inputs. It must accept specific arguments and return a dictionary containing the proposed next test condition.

```python
from ax.service.ax_client import AxClient

def my_custom_planner_logic(parameters: list[dict], 
                            objective: dict, 
                            constraints: list[str], 
                            data: list[dict],
                            settings: dict) -> dict:
    """
    Args:
        parameters: List of Ax formatted parameters
        objective: Dict containing the ObjectiveProperties object
        constraints: List of string constraints (evaluated by SymPy)
        data: List of dicts representing historical trials ('parameters' and 'objectives')
        settings: Dict of planner settings from ARES OS
    """
    # 1. Initialize your Ax Client
    ax_client = AxClient()
    ax_client.create_experiment(
        parameters=parameters,
        objectives=objective,
        parameter_constraints=constraints
    )

    # 2. Attach historical and seed data
    for trial in data:
        _, trial_index = ax_client.attach_trial(parameters=trial['parameters'])
        ax_client.complete_trial(trial_index=trial_index, raw_data=trial['objectives'])
        
    # 3. Generate the next trial
    parameterization, _ = ax_client.get_next_trial()
    
    # 4. Return the predicted parameters
    return parameterization
```

By adhering to this structure, your custom Ax planner will automatically inherit robust data parsing, smart seeding, and seamless integration with the ARES OS environment.

## License
This project is licensed under the MIT License. Copyright (c) 2026 AFRL-ARES.