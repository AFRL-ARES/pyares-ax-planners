#!/usr/bin/env python3
# -*- coding:utf-8 -*-
###
# File: /src/pyares_bo/ax/sobo_continuous_ax.py
# Project: pyares-bo-planners
# Created Date: Friday, March 13th 2026, 2:18:34 pm
# Author(s): Arthur W. N. Sloan
# -----
# MIT License
# 
# Copyright (c) 2026 AFRL-ARES
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# 
###
from ax.service.ax_client import AxClient, ObjectiveProperties
from PyAres import PlanRequest, PlanResponse
from pathlib import Path
import pandas as pd
import warnings
from datetime import datetime

smart_seed_conditions = list()
def sobo_cont_planner(request: PlanRequest) -> PlanResponse:
    '''
    Top level sobo_cont_planner function handles partsing the plan request into values useful for the planner
    Based on the user settings, the planner will accept seed data, generate seed points or procede straight to planning using Ax
    '''
    global smart_seed_conditions
    # PyARES compatible single objective bayesian optimization planner based on the default Ax behavior
    # Initilizes a new Ax API client for each request 

    # Parse out settings. If settings are not defined in the request ocming over from ARES OS the fields don't exist 
    # So there's a bit of checking
    if 'Constraints' in request.settings:
        constraints = request.settings['Constraints']
    else:
        constraints = list()
    
    if "Seed Data" in request.settings:
        seed_data = request.settings['Seed Data']
    else:
        seed_data = ''
    if "Minimize" in request.settings:
        minimize = request.settings['Minimize']
    else:
        minimize=False
    if 'Smart Seed' in request.settings:
        smart_seed_enabled = request.settings['Smart Seed']
    else:
        smart_seed_enabled = False
    
    if 'Verbose Output' in request.settings:
        verbose = request.settings['Verbose Output']
    else:
        verbose = False
    
    parameters = [{'name':p.name,
                    'type':'range',
                    'bounds':[p.minimum_value, p.maximum_value]} for p in request.parameters]
    objective = {'objective':ObjectiveProperties(minimize=minimize)}
    start = datetime.now()
    N_trial = len(request.analysis_results) # How many trials have been completed by the planner
    print(f"--- Planning Trial #{N_trial} ---")
    print(f'Planning Started at: {start.strftime("%Y-%m-%d %H:%M:%S")}')

    
    
    # Check if seed data is provided by the user
    if seed_data is not None and seed_data != '':
        # Process The Seed Data into a form that can be passed to the planner
        seed_data_list = process_seed_data(seed_data,parameters)
        has_seed_data = True
    else:
        seed_data_list = list()
        has_seed_data = False
        
    if verbose:
        if has_seed_data:
            print(f'\t{len(seed_data_list)} seed data points found')
        else:
            print('\tNo seed data provided')

    # Process the data coming in with the planning request
    if N_trial > 0:
        trial_data = process_experimental_data(request)
    else:
        trial_data = []

    if has_seed_data:
        hist_data = seed_data_list + trial_data
    else:
        hist_data = trial_data
    
    # If seed data is not provided, check if the smart_seed setting is enabled
    if not has_seed_data and smart_seed_enabled == True:
        seed_type = request.settings['Smart Seed Type']
        smart_seed_points = request.settings['Number of Seed Experiments']
        # We want to be generic when supporting seeding methods, so we'll only generate them once
        # That way seed generation is determinisitc 
        if len(smart_seed_conditions) == 0:
            if verbose:
                print(f'\tSmart seed enabled - gererating {smart_seed_points} seed points of type "{seed_type}"')
            smart_seed_conditions = smart_seed(parameters, seed_type, smart_seed_points,constraints)


    if len(smart_seed_conditions) !=0 and N_trial <= len(smart_seed_conditions):
        if verbose:
            print(f'\tSeed Collection Enabled: Acquiring seed data point {N_trial}/{len(smart_seed_conditions)}')
        # If there are planner-generated seed conditions, we don't call the planner and just return the seed conditon
        parameterization = smart_seed_conditions[N_trial]
    else:
        # Pass data to the ax planner and get the next point:
        if verbose:
            print(f'\tUsing {len(hist_data)} data points (seed + trials) for BO planning')
        parameterization = sobo_planner(parameters,objective,constraints,hist_data)

    # Repackage the parameterization for an ARES Plan Response
    parameter_names = list(parameterization.keys())
    new_test_condition = list(parameterization.values())
    print("\tProposed test condition:")
    for n,v in zip(parameter_names, new_test_condition):
        print(f"\t{n} = {v:.3f}")
    end = datetime.now()
    delta= end-start
    print(f"--- End Planning for Trial #{N_trial} (Took {delta.total_seconds()}) seconds---")

    return PlanResponse(parameter_names=parameter_names, parameter_values=new_test_condition)


def process_seed_data(seed_data_path:str, parameters:list[dict]) -> list[dict]:
    seed_data = Path(seed_data_path)
    parameter_names = [i['name'] for i in parameters]


    if not seed_data.exists():
        raise Exception(f'Seed data file "{str(seed_data)}" does not exist')
    if not seed_data.is_file():
        raise Exception(f'Seed data file "{str(seed_data)}" is not a file')
    
    ext = seed_data.suffix
    if ext == '.xlsx' or ext == '.xls':
        df = pd.read_excel(str(seed_data))
    elif ext == 'csv':
        df = pd.read_csv(str(seed_data))
    else:
        raise Exception('Seed Data file could not be read. File should be an Excel file (.xls, .xlsx) or .csv')


    if not all([i in df.columns for i in parameter_names]):
        raise Exception(f'Could not match all planner parameter names to column headers (case senesitive). \
                            Planner Parameters: {parameter_names} Column Names: {df.columns}')
    
    if 'objective' not in df.columns:
        raise Exception(f'Could not find a column named "objective" in column headers. Column Names: {df.columns}')
    
    if not all([i in parameter_names + ['objective', 'Index', 'index'] for i in df.columns]):
        warnings.warn('Extra columns were found in the seed data file that were not used in planning.')

    # Format the data in to list of dicts format that the planner expects
    data = list()
    for i in range(len(df)):
        obj_dict = {'objective':df['objective'][i]}
        par_dict = {p:df[p][i] for p in parameter_names}
        data.append({'parameters':par_dict,
                     'objectives':obj_dict})
    return data

    
def process_experimental_data(request: PlanRequest) -> list[dict]:
    data = list()
    for i in range(len(request.analysis_results)):
        # Make the dict of parameter:value pairs
        # The first parameter history entry will always be blank as it is called first so add one to the index
        par_dict = {p.name:p.param_history[i+1].achieved_value for p in request.parameters}
        obj_dict = {'objective':request.analysis_results[i]}
        data.append({'parameters':par_dict,
                     'objectives':obj_dict})
    return data

def sobo_planner(parameters:list[dict], 
                 objective:dict, 
                 constraints:list[str], 
                 data:list[dict]) -> dict:
    """Initilaizes and attaches previous trials to 

    Args:
        parameters (list[dict]): List of Ax formatted parameters
        objective (dict): Dict containing a single entry for the objective, an Ax. ObjectiveProperties object.
        constraints (list[str]): List of Ax planning constraints, should be an array of strings that can be evaluated by sympy with 
                                 variables that match the parameter names 
        data (list): A list of dicts corresonding to previous trials with fields named 'parameters' and 'objectives'

    Returns:
        dict: The paramters of the new trial, prediced by the BO planner
    """

    # Function to actually plug everything into the Ax API client
    ax_client = AxClient()
    try:
        ax_client.create_experiment(parameters=parameters,
                                    objectives=objective,
                                    parameter_constraints=constraints)
    except Exception as e:
        raise Exception(f'Error Creating the Ax API client: {e}')

    if len(data) > 0:
        for i in range(len(data)):
            # Make the dict of parameter:value pairs
            params = data[i]['parameters']
            obj_score = data[i]['objectives']

            _, trial_index = ax_client.attach_trial(parameters=params)
            ax_client.complete_trial(trial_index=trial_index, raw_data=obj_score)
    
    parameterization, _ = ax_client.get_next_trial()
    return parameterization

def smart_seed(parameters: list[dict], seed_type:str, N_seed_ponts: int, constraints:list[str] | None) -> list:
    if seed_type == 'Latin Hyper Cube':
        from seed_methods import LHSMDU_Generator
        cube_gen = LHSMDU_Generator(parameters,N_seed_ponts)
        if constraints is not None and len(constraints) > 0:
            seed_df = cube_gen.make_constrained_hypercube(constraints)
        else:
            seed_df = cube_gen.make_hypercube()

    #convert the dataframe from the seed generator to the list of dicts format
    seed_conditions = list()
    for i in range(len(df)):
        entry_dict = {k:seed_df[k][i] for k in seed_df.columns}

        seed_conditions.append(entry_dict)

    return seed_conditions
