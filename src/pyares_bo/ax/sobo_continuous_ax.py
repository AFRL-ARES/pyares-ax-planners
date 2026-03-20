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

smart_seed_conditions = list()
def sobo_cont_planner(request: PlanRequest) -> PlanResponse:
    '''
    Top level sobo_cont_planner function handles partsing the plan request into values useful for the planner
    Based on the user settings, the planner will accept seed data
    '''
    global smart_seed_conditions
    # PyARES compatible single objective bayesian optimization planner based on the default Ax behavior
    # Initilizes a new Ax API client for each request 

    # Top Level 
    constraints = request.settings['Constraints']
    seed_data = request.settings['Seed Data']
    minimize = request.settings['Minimize']
    smart_seed = request.settings['Smart Seed']

    parameters = [{'name':p.name,
                    'type':'range',
                    'bounds':[p.minimum_value, p.maximum_value]} for p in request.parameters]
    objective = {'objective':ObjectiveProperties(minimize=minimize)}

    N_trial = len(request.analysis_results) # How many trials have been completed by the planner
    

    # Check if seed data is provided by the user
    if seed_data is not None or seed_data is not '':
        has_seed_data = True
        # Process The Seed Data into a form that can be passed to the planner
        seed_data = process_seed_data(seed_data,parameters)
    else:
        has_seed_data = False

    # If seed data is not provided, check if the smart_seed setting is enabled
    if not has_seed_data and smart_seed == True:
        seed_type = request.settings['Smart Seed Type']
        smart_seed_points = request.settings['Number of Seed Experiments']

        # We want to be generic when supporting seeding methods, so we'll only generate them once
        # That way seed generation is determinisitc 
        if len(smart_seed_conditions) == 0:
            smart_seed_conditions = smart_seed(parameters, seed_type, smart_seed_points,constraints)


        if N_trial < smart_seed_points and len(smart_seed_conditions) == 0:
            pass
        else:
            parameterization = smart_seed_conditions[N_trial]
    

    # Process the data coming in with the planning request


    trial_data = process_experimental_data(request)
    # Process previous trials for passing to planner
    # Merge data with seed data (if any)

    # Pass data to the ax planner and get the next point:
    parameterization = sobo_planner(parameters,objective,constraints,hist_data)




    parameter_names = list(parameterization.keys())
    new_test_condition = list(parameterization.values())
    print("\tProposed test condition:")
    for n,v in zip(parameter_names, new_test_condition):
        print(f"\t{n} = {v:.3f}")
    print(f"--- End Planning for Trial #{N_trials}---")

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
    for i in range(request.analysis_results):
        # Make the dict of parameter:value pairs
        par_dict = {p.name:p.param_history[i].achieved_value for p in request.parameters}
        obj_dict = {'score':request.analysis_results[i]}
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

            trial_index = ax_client.attach_trial(parameters=params)
            ax_client.complete_trial(trial_index=trial_index, raw_data=obj_score)
    
    parameterization, _ = ax_client.get_next_trial()
    return parameterization

    

    
    









    # If seed data is provided, organize it into dicts that are easy to pass to the function wrappign the planner
    N_trials = len(request.analysis_results)
    # Organize Data 




    if N_trials == 0:
        pass
    else:




    ax_client = AxClient()
    parameters = [{'name':p.name,
                    'type':'range',
                    'bounds':[p.minimum_value, p.maximum_value]} for p in request.parameters]
    parameter_names = [i['name'] for i in parameters]
    objective = {'score':ObjectiveProperties(minimize=minimize)}
    ax_client.create_experiment(parameters=parameters,
                                objectives=objective,
                                parameter_constraints=constraints)

    # Placeholder seed data handling
    # TODO: Update once this part of ARES OS matures
    '''
    Seed Data Handling:
    Planner checks to see if the seed data setting has been assigned and then checks that the seed data is a file and tries to read it
    Seed data should be either an Excel file or a csv with column names that match the parameters being used
    '''
    use_seed = False
    seed_data = request.settings['Seed Data']
    if seed_data is not None or seed_data is not'':
        seed_data = Path(seed_data)
        if seed_data.is_file():
            ext = seed_data.suffix
            if ext == '.xlsx' or ext == '.xls':
                data = pd.read_excel(str(seed_data))
            elif ext == 'csv':
                data = pd.read_csv(str(seed_data))
            else:
                use_seed = False
                raise Warning('Seed Data file could not be read. File should be an Excel file (.xls, .xlsx) or .csv, proceding without seed data')

            # Data integrrity checks
            # Check that all parameters are present
            if not all([i in data.columns for i in parameter_names]):
                use_seed = False
                raise Warning('Could not find all experimental parameters in the seed data, proceding without seed data')
            if 'objective' not in data.columns:
                use_seed = False
                raise Warning('Could not locate "objective" column in seed data, proceding without seed data')
            

        else:
            use_seed = False
            raise Warning('Seed Data file does not exist, proceding without seed data')
                
                

            # Check that an 'objective' column is present

            # check if there are unused columns. If so warn the user
    else:
        use_seed = False  


    if use_seed:
        for i in range(N_seed_points):
            # Make the dict of parameter:value pairs
            params = {name:data[name].to_numpy()[i] for name in parameter_names}
            obj_score = {'score':data['objective'].to_numpy()[i]}
            trial_index = ax_client.attach_trial(parameters=params)
            ax_client.complete_trial(trial_index=trial_index, raw_data=obj_score)

    N_trials = len(request.analysis_results)
    print(f'--- Planning Trial #{N_trials} ---')
    if N_trials == 0: # First run
        parameterization, _ = ax_client.get_next_trial()
        # If any of the parameters have a specified initial value overwrite the planner suggestion
        for p in request.parameters:
            if isinstance(p.initial_value, float):
                print(f'\tInitial value found, overriding planner: {p.name}: p.initial_value')
                parameterization[p.name] = p.initial_value
    
    else: # attach data from previous experiments to the experiment so it can plan the requested point.
        print(f'\t Found data for {N_trials} previous experimental data points')
        for i in range(N_trials):
            # Make the dict of parameter:value pairs
            params = {p.name:p.param_history[i].achieved_value for p in request.parameters}
            obj_score = {'score':request.analysis_results[i]}

            trial_index = ax_client.attach_trial(parameters=params)
            ax_client.complete_trial(trial_index=trial_index, raw_data=obj_score)
        
        parameterization, _ = ax_client.get_next_trial()

    parameter_names = list(parameterization.keys())
    new_test_condition = list(parameterization.values())
    print("\tProposed test condition:")
    for n,v in zip(parameter_names, new_test_condition):
        print(f"\t{n} = {v:.3f}")
    print(f"--- End Planning for Trial #{N_trials}---")

    return PlanResponse(parameter_names=parameter_names, parameter_values=new_test_condition)

def smart_seed(parameters: list[dict], seed_type:str, N_seed_ponts: int) -> list:
    
    return seed_conditions