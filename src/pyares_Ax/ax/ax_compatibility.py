#!/usr/bin/env python3
# -*- coding:utf-8 -*-
###
# File: /src/pyares_Ax/ax/ax_compatibility.py
# Project: pyares-ax-planners
# Created Date: Tuesday, March 31st 2026, 11:11:08 am
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

from PyAres import PlanRequest, PlanResponse, AresDataType, Outcome, AresPlannerService, Plan, PlannedParameter
from pathlib import Path
import pandas as pd
import warnings
from datetime import datetime
from ax.service.ax_client import ObjectiveProperties
import logging 
from typing import Any, Callable, Optional, Union, Dict
from PyAres.Models import AresSchemaEntry, Limits

import sympy as sp
import re
import io
import os
from ..visualization import BokehIterativeVisualizer
from bokeh.server.server import Server
import threading

class PyAres_Ax_Planner(object):
    """
        Generic parent class for wrapping planning routines that utilize meta's Ax API in a PyAres compatibility layer
    """
    def __init__(self):
        # Initialize generic Planner attributes
        self.settings : dict = {}
        self.settings_list = []
        self.achieved_values = {}
        self.planned_values = {}
        self.seed_data = []
        self._ares_parameters:list = []
        self._planner_parameters:list = []
        self._implicit_parameters:list =[]
        self._implicit_values:list = []
        self.parameters = []
        self.implicit_parameters = []
        self.derived_values = []
        self._seed_data_df: pd.DataFrame = pd.DataFrame()
        self._planned_df: pd.DataFrame = pd.DataFrame()
        self._acheived_df: pd.DataFrame = pd.DataFrame()
        # Place Holders for for planner specific information, these should be overridden by child classes
        self.plan_function : Callable = plan_function
        self.planner_input_type : AresDataType = AresDataType.NUMBER
        self._visualizer = None
        self._visualizer_server = None
        self.visualizer_port:int = 5555 # default to 5555

        # Generic Planner Info, expected to be overridden by child classes
        self.name : str = 'PyAres Prototype Ax Planner'
        self.version_number : str = 'x.x.x'
        self.description : str = "Parent class for making Ax API based planners compatible with ARES OS though PyARES"

        # Generic planner settigns, can be modified or overridden by child classes
        '''
        Defualt Parameters that  will be a part of all planners
        'Seed Data': A string that points to a file (.csv, .xls, .xlsx) containing seed data. This setting will change to support more file types and as 
                    Ares OS's handling of seed data evolves.
        '''
        # TODO: Update Seed Data Handling when ARES OS data handling is updated
        self.add_setting(setting_name='Seed Data',
                         setting_type=AresDataType.STRING,
                         description="Path to a .csv or excel file of seed data for the planner"
                         )
        self.add_setting(setting_name='Constraints',
                         setting_type=AresDataType.STRING_ARRAY,
                         description='Mathematical constraints for the planner to apply expressed as a list of SymPy interperable strings'
                         )
        self.add_setting(setting_name='Implicit Values',
                         setting_type=AresDataType.STRING_ARRAY,
                         description='Implicit/derived values that the planner needs to calculate constraints expressed as a list of SymPy interperable strings'
                         )
        self.add_setting(setting_name='Verbose Output',
                         setting_type=AresDataType.BOOLEAN,
                         default_value=True,
                         description='Increases the level of ouput in the service terminal, useful for debugging issues'
                         )
        # TODO: This only allows for slecting on option or the other. Do we epect the possibility of a mix?
        self.add_setting(setting_name='Parameter Value Type',setting_type=AresDataType.STRING,optional=False,constraints=['Planned','Acheived'],default_value='Planned')
        # NOTE: Once the visualizer service is added this will probably need to be removed/moved
        self.add_setting(setting_name="Output Folder",setting_type=AresDataType.STRING,optional=False,default_value=str(os.path.expanduser('~')))

    ### Interface functions ###
    # These functions are expected to be implemented in all planner classes and are the primary way PyAres interfaces with the planner
    def info(self) -> dict:
        """
        Returns basic information about the planner that will be repored to ARES OS.
        Storing it in the planner class makes it easier to keep a single source of truth for these values
        """
        return {'name':self.name,'version':self.version_number,'description':self.description}
    

    def add_setting(self, 
                    setting_name: str, 
                    setting_type: AresDataType,
                    default_value: Any = None,
                    optional: bool = True,
                    constraints: Union[list, None] = None,
                    struct_schema: Optional[Dict[str, AresSchemaEntry]] = None,
                    list_element_schema: Optional[AresSchemaEntry] = None,
                    limits: Optional[Limits] = None,
                    description: Optional[str] = None) -> None:
        """Add a setting to the planner

        Args:
            setting_name (str): name string for the planner
            setting_type (AresDataType): AresDataType for the setting
            optional (bool, optional): if the setting is opitonal. Defaults to True.
            constraints (list, optional): list of allowed values for the setting. Defaults to [].
            default_value (Any, optional): Default value for the planner. Defaults to ''.
        """
        self.settings_list.append({'setting_name':setting_name, 
                                   'setting_type':setting_type,
                                   'optional':optional,
                                   'constraints':constraints,
                                   'default_value':default_value,
                                   'struct_schema':struct_schema,
                                   'list_element_schema':list_element_schema,
                                   'limits':limits,
                                   'description':description
                                   })


    def configure_settings(self, planner:AresPlannerService) -> AresPlannerService:
        '''
        Applies the planner settings to the AresPlannerService object. Also initializes default settings within the
        planner object's own internal settings atrribute.
        Note: 
        This initializes the internal settings atribute with default values so that any check of those settings 
        do not have to account for the possibly of a missing value. This behavior may change when PyAres messages
        are updated to support default values.
        '''
        planner.add_supported_type(self.planner_input_type)
        for s in self.settings_list:
            self.settings[s['setting_name']] = s['default_value']
            _s = dict(s)
            planner.add_setting(_s.pop('setting_name'), _s.pop('setting_type'),**_s)
        return planner
    

    def call_planner(self,request: PlanRequest) -> list[Plan]|PlanResponse:
        """
        Generic planner calling function that manages sorting all of the data contained in the incoming
        plan request object into a format that is more compaitble with how Ax-based planners want their data.

        Currently this model assumes that the planner itself is not tracking the overall state of the planning, and the Ax API client
        is re-constructed each call. This has the benefit of making it possible to call out to a single planner service from multiple instances of
        ARES OS. THis means that the data for all trials is provided with every call to the planner function

        This function also manages initial conditions that may be provided by ARES OS for the first run, allowing some or all parameters returned by the planner 
        to be overridden

        This function manages the use and calculation of implicit values and implicit parameters, this allows planning over different values than
        those that are provided/expected by ARES OS. 
        Examples:
            -The total flow rates for N components must remain constant, so planning occurs over N-1 components, with the final value calcuated 
             from the specified total and the sum of the planned variables


        Eventually we would like to be able to support the use of planning parameters that are based on the difference, ratio, etc. of two or more variables,
        or support planning over the log, or exponentiation of the raw values coming from ARES OS. but this leads to some issues of creating degenerate parmeter spaces
        as well as degenerate solutions for parameter minimimus and maximums, and so this is not supported at this time.

        
        Args:
            request (PlanRequest): Incoming PyAres PlanRequest object from ARES OS

        Returns:
            PlanResponse: PyARES PlanResponse object to be sent back to ARES OS. Contains parameter names,
                            values and a status message for the planning process.
        """
        #TODO: Evaluate if it is desierable for to support the use of a Persistent Ax client an what needs to change to do that.

        # Start planning, track elapsed time
        start = datetime.now()
        self._config_ouptut(request)
        self.N_trial = len(request.analysis_results)
        self.trial_range = list([i+self.N_trial for i in range(request.batch_size)])
        if request.batch_size == 1:
            print(f"--- Planning Trial #{self.N_trial} ---")
        else:
            print(f"--- Planning Trial #{self.trial_range[0]} - #{self.trial_range[-1]} (Batch Size = {request.batch_size})---")
                    
        print(f' Planning Started at: {start.strftime("%Y-%m-%d %H:%M:%S")}')
        # Updates settings from the request and parses a few essential values
        self.settings.update(request.settings)
        self.verbose = self.settings['Verbose Output']
        self.constraints = self.settings['Constraints']

        # Configure Planning parameters and objectives
        self._configure_parameters(request)
        self._configure_objectives(request)
        self._configure_constraints()

        # Parse seed data and previous trials into a format that is useful for the planner
        self._process_seed_data()
        if self.verbose:
            print(f'\tFound {len(self._seed_data_df)} points of existing seed data')
        self._process_experimental_data(request)
        if self.verbose:
            print(f'\tFound {self.N_trial} points of existing experimental data')
        buffer, handler, logger = self._ax_log_interceptor()

        if self.N_trial == 0:
            # Separate logic to handle the first run of the planner for things specified initial conditions
            ares_response, outcome, override_flags = self._plan_first_run(request)
        else:
            override_flags = [False for i in self._ares_parameter_names]
            try:
                # NOTE: With batch planning response from the plan function will be a nested dict with top level
                #       keys corresponding to the iteration number and each entry represning the trial dict
                plan_response = self.plan_function(parameters=self._planner_parameters,
                                                objectives=self.objectives,
                                                constraints=self.constraints,
                                                data=self.data,
                                                settings=self.settings,
                                                batch_size=request.batch_size)
                outcome= Outcome.SUCCESS
            except Exception as e:
                print(f' Planning Failed with error:{e}')
                plan_response = {n:{p:-1 for p in self.parameter_names} for n in self.trial_range}
                outcome = Outcome.FAILURE
            ares_response = self._convert_plan_to_ares(plan_response)

        # Capture Ax output for verbose output setting
        captured_text = buffer.getvalue()
        if self.verbose:
            print("\n~~~ Begin Ax logs ~~~")
            print(captured_text.replace('\n','\n\t'))
            print("~~~ End Ax Ax logs ~~~\n")
        # Clean up when done
        logger.removeHandler(handler)
        buffer.close()

        # Print new conditions to terminal output
        print(" Proposed test condition(s):")
        for i in self.trial_range:
            if request.batch_size > 1:
                print(f" ~~~ Trial # {i}: ~~~")
            for j,(n,v) in enumerate(zip(ares_response[i].keys(), ares_response[i].values())):
                if override_flags[j] and self.N_trial == 0 and i == 0 :
                    print(f"\t{n} = {v:.3f} (Overriden by supplied inital value)")
                else:
                    print(f"\t{n} = {v:.3f}")

        end = datetime.now()
        delta= end-start
        if request.batch_size == 1:
            print(f"--- End Planning for Trial #{self.N_trial} (Took {delta.total_seconds()}) seconds---")
        else:
            print(f"--- End Planning for Trials #{self.trial_range[0]} - #{self.trial_range[-1]} (Took {delta.total_seconds()}) seconds---")

        plan_list = []
        for n in self.trial_range:
            batch_item = ares_response[n]
            param_list = [PlannedParameter(parameter_name=k,parameter_value=v) for (k,v) in zip(batch_item.keys(),batch_item.values())]
            plan_list.append(Plan(planned_parameters=param_list,outcome=outcome))

        # response = PlanResponse(parameter_names=list(ares_response.keys()),
        #                         parameter_values=list(ares_response.values()),
        #                         outcome=outcome)
        if self.N_trial >= 1:
            # If the Bokeh visualizer hasn't been started yet, start it, otherwise, update it
            if self._visualizer is None:
                self._visualizer = BokehIterativeVisualizer(self._ares_parameter_names,
                                                            self.objective_names,
                                                            {k:self.objectives[k].minimize for k in self.objectives},
                                                            {item['name']:item['bounds'] for item in self._ares_parameters}
                )
                self._visualizer_server = Server({"/": self._visualizer.bkapp}, port=self.visualizer_port,allow_websocket_origin=["*"])
                self._visualizer_server.start()
                io_thread = threading.Thread(target=self._visualizer_server.io_loop.start)
                io_thread.daemon = True
                io_thread.start()

            self._visualizer.push_update(self.data_df)
            self._visualizer.save_snapshot(str(self.settings['_exp_output_dir']))

        # plot_trials_progress(request)
        return plan_list
    
    ### Support functions - Not intended for general interfacing
    ##Override these functions when configuring your planner subclass

    def _config_ouptut(self,request: PlanRequest):
        output_folder = request.settings['Output Folder']
        campaign_name = request.request_metadata.campaign_name
        n_iter = len(request.analysis_results)
        experiment_time = datetime.strptime(request.request_metadata.experiment_start_time, "%Y-%m-%d %H:%M:%S")
        experiment_date = experiment_time.strftime("%Y-%m-%d") # Just getting the YMD to put in the name for easy sorting
        experiment_time = experiment_time.strftime("%Y-%m-%dT%H-%M")

        experiment_name = f'{experiment_time}_experiment_{n_iter}'
        # experiment_id = request.request_metadata.experiment_id

        write_folder = Path(output_folder)/(experiment_date +'_'+campaign_name)/experiment_name
        write_folder.mkdir(exist_ok=True,parents=True)
        self.settings['_exp_output_dir'] = write_folder

    def _configure_objectives(self,request: PlanRequest):
        """
        Configures the objectives for the Ax planner. This function is expected to be overridden in the child class to propperly set the planning goals.

        """

        self.objectives = {'objective':ObjectiveProperties(minimize=False)} 

    ## General support functions, may be overridden if necessary but shouldn't need to be for most use cases
    def _configure_parameters(self,request: PlanRequest):
        '''
        This function pasrses input parameters, constraints, and implicit values to figure out what to pass to the planner routine as well as 
        building the definitions for the translation layer for calculating implicit parameters

        The logic of this process is as folows:
        1. Establish ARES Parameters
            Reads the parameters from the PlanRequest message and stores them in self._ares_parameters as a list of Ax compatible parameter definition dicts.

        If there are no entries in self.settings["Implicit Values"]  then we're done and self._planning parameters = self._ares_parameters, otherwise:
        2. Before dealing with derived planner parameters we need to define any implicit values that have been provided
            -There are two classes of calculated implicit values: implicit parameters, which correspond to an ARES Parameter, and implict values, which 
                could be used to calculate an implicit parameter or be used as a planning parameter
            -Implicit values should be defined using sympy compatible strings "<implicit value> = <expression>" 
        3. Which entries define implicit parameters and which define implicit values is automatically determined by comparing the name of the 
            value to the parameter names received from ARES OS, so it is essential that these match exactly.
        
        
        Example: The flow total rate of 4 reactants must remain constant, Ares sends over the parameters flow_1, flow_2, flow_3, and flow_4
            self.settings['Constraints] = ["flow_1 + flow_2 + flow_3 <= {total_flow}"]
            self.settings["Implicit Values"] = ['flow_4 = total_flow - (flow_1 + flow_2 + flow_3), 'total_flow = 100']

        In this example, the constraint string will be evaluated prior to being passed to the planner

        '''
        # TODO: Reevaluate logic down the road if flagging parameters as implicit is supported as part of ARES planning requests
        # TODO: Update parameter list consruction once ARES OS requests support categorical variables
        # TODO: Check on sympy input sanitization
        self._ares_parameters = [{'name':p.name, 
                                'type':'range',  
                                'bounds':[p.minimum_value, p.maximum_value]} for p in request.parameters]
        
        if len(self.settings['Implicit Values']) == 0:
            self._planner_parameters = self._ares_parameters
        else:
            for value_str in self.settings['Implicit Values']:
                # Split out the paramter strings
                expr_elements = value_str.split('=')
                lhs = expr_elements[0].strip()
                rhs = expr_elements[-1].strip()
                # evaluate RHS to sympy expression then turn it into a function
                expr = sp.sympify(rhs) 
                symbols = list(expr.free_symbols)
                symbol_names = [symbol.name for symbol in symbols]
                func = sp.lambdify(symbols, expr, modules='numpy')
                # Make a record of what the value name is, the evaluation function and what inputs (in order) go into it.
                value_dict = {'name':lhs, 'function':func, 'inputs':symbol_names,'expression':value_str}
                # If the parameter is an ares parameter it gets stored in self._implicit_parameters, otherwise it gets stored in self._implicit_values
                if lhs in self._ares_parameter_names: 
                    self._implicit_parameters.append(value_dict)
                else:
                    self._implicit_values.append(value_dict)
                #sets the values that the planner will actuall operate on
                self._planner_parameters = [p for p in self._ares_parameters if p['name'] not in self._implicit_parameter_names]                
        # TODO: Figure out how to handle telling the planner to plan over derived variables rather than the parameters recieved from ARES OS and how to solve 
        # the degeneracy and bounds issues that come with that capibility


    def _configure_constraints(self):
        """
        Supports the use of implicit values in constraints so that variables can be defined once and reused 
        """
        if len(self.constraints) >0:
            for i, con_str in enumerate(self.constraints):
                vars_to_evaluate = re.findall(r'\{([^}]+)\}', con_str) # Finds things inside curly braces
    
                if len(vars_to_evaluate)>0:
                    computed_values = {}
                    for var in vars_to_evaluate:
                        # We only want to compute it once if it appears multiple times
                        if var not in computed_values:
                            val = self._eval_implicit({},var)
                            computed_values[var] = val
                            
                    # 3. Format the original string using the computed dictionary
                    formatted_str = con_str.format(**computed_values)
                    self.constraints[i] = formatted_str

    def _process_experimental_data(self, request: PlanRequest):
        """
        Processes the experimental trial data in the PyAres request into a pandas dataframe for easier manipulation and reshaping .

        Args:
            request (PlanRequest): PyAres Planning request
        """
        
        cols = self._ares_parameter_names + self.objective_names
        planned_df = pd.DataFrame(columns=cols)
        achieved_df = pd.DataFrame(columns=cols)
        for p in request.parameters:
            planned_df[p.name] = [p.param_history[i].planned_value for i in range(len(request.analysis_objectives))]
            achieved_df[p.name] = [p.param_history[i].achieved_value for i in range(len(request.analysis_objectives))]

        # NOTE: Not the cleanest way to do this, but ultimately i think a better way to go about it is to make a helper function in PyAres
        #       that provies the objective values in a friendlier format.
        objective_values_dict = {o:[] for o in self.objective_names}
        for iter in request.analysis_data:
            for objective in iter.analysis_objectives:
                if objective.objective_name in self.objective_names:
                    objective_values_dict[objective.objective_name].append(objective.objective_value)

        for o in self.objective_names: 
            planned_df[o] = objective_values_dict[o]
            achieved_df[o] = objective_values_dict[o]
        
        self._acheived_df = achieved_df
        self._planned_df = planned_df

    def _process_seed_data(self):
        """
        Read in Seed data, this is a top level function to call data type specific import functions

        At the moment we only support reading data in from files.

        Note that the objective value score in the seed data file must have the column name 'objective' (case sensitive)
        
        """
        seed_as_path = Path(self.settings['Seed Data'])
        # TODO: Reevaluate once there are more data curation tools availible to ARES OS
        # TODO: Update instructions once multi-objective planning is implemented.
        if self.settings['Seed Data'] != '':
            if seed_as_path.is_file():
                self._seed_data_df = self._process_seed_data_file(seed_as_path)
            else:
                warnings.warn(f"Seed data setting {self.settings['Seed Data']} is not recognized as a file. Data will not be used")
                self._seed_data_df = pd.DataFrame()
        else:
            self._seed_data_df = pd.DataFrame()


    def _process_seed_data_file(self,file:Path) ->pd.DataFrame:
        """
        Formats the data in a .csv or excel file to the list of dicts format required by the planner. With checks to ensure that the required values are present.

        Args:
            file (Path): Path to the csv or excel file to read

        Returns:
            data (list): Formatted data
        """
        ext = file.suffix
        if ext == '.xlsx' or ext == '.xls':
            df = pd.read_excel(str(file))
        elif ext == 'csv':
            df = pd.read_csv(str(file))
        else:
            raise Exception('Seed Data file could not be read. File should be an Excel file (.xls, .xlsx) or .csv')
        
        if not all([i in df.columns for i in self.parameter_names]):
            raise Exception(f'Could not match all planner parameter names to column headers (case senesitive). \
                                Planner Parameters: {self.parameter_names} Column Names: {df.columns}')
        if 'objective' not in df.columns:
            raise Exception(f'Could not find a column named "objective" in column headers. Column Names: {df.columns}')
        
        if not all([i in self.parameter_names + ['objective', 'Index', 'index'] for i in df.columns]):
            warnings.warn('Extra columns were found in the seed data file that were not used in planning.')
        
        return df

    def _eval_implicit(self,input_dict:dict, target_name:str):
        """
        Calculates the target implicit value or factor based on inputs and definitions.
        Args:
            input_values (dict): A dictionary of explicit inputs (e.g., {'x1': 5, 'x2': 10}).
            target_name (str): The name of the parameter/factor to compute.
            
        Returns:
            The evaluated result for the target_name.
        """
        implicit_params = self._implicit_parameters
        derived_vals = self._implicit_values

        
        # 1. Combine computable formulas into a single registry for O(1) lookups
        computables = {}
        for param in implicit_params:
            computables[param['name']] = param
        for value in derived_vals:
            computables[value['name']] = value
            
        # 2. Initialize a cache with our base explicit inputs
        # We copy the dictionary to avoid mutating the original input dict
        cache = input_dict.copy()
        
        # Track nodes we are currently evaluating to catch circular dependencies
        visiting = set()
        
        # 3. Define the recursive solver
        def resolve(name):
            # Base case: we already know the value (it was input or already computed)
            if name in cache:
                return cache[name]
                
            # Cycle detection: if we see the same variable while already trying to compute it
            if name in visiting:
                raise ValueError(f"Circular dependency detected involving: '{name}'")
                
            # Error handling: if a variable is entirely missing from inputs and formulas
            if name not in computables:
                raise KeyError(f"Missing definition or input for variable: '{name}'")
                
            # Mark as currently visiting
            visiting.add(name)
            
            # Extract the rule
            rule = computables[name]
            func = rule['function']
            dep_names = rule['inputs']
            
            # Recursively resolve all arguments needed for this function
            resolved_args = [resolve(dep) for dep in dep_names]
            
            # Calculate the result
            result = func(*resolved_args)
            
            # Cache the result so we don't have to compute it again
            cache[name] = result
            visiting.remove(name)
            
            return result
        # 4. Trigger the recursive evaluation for our target
        return resolve(target_name)
    

    def _plan_first_run(self,request: PlanRequest) -> tuple[dict,Outcome,list]:
        response = []

        initial_conditions = self._get_initial_conditions(request)
            # If all parameters are present in the initial conditions response, skip planning
            # Planners could be aware of their own history so we don't want to call the planner unecessarily.
        if all(p in initial_conditions.keys() for p in self._ares_parameter_names):
            plan_response = initial_conditions
            initial_condition_override = [True for i in self._ares_parameter_names]
            outcome = Outcome.SUCCESS
        else:
            initial_condition_override = [False for i in self._ares_parameter_names]
        # If some or all values do not have an initial value specified we still need to get values from the planner
        # and overwrite the ones that we have initial values for.
        
        # NOTE: This could cause some weird behavior if the planner is state aware and tracking previous values, since we're overwriting what it is sending back without telling it.

            try:
                plan_response = self.plan_function(parameters=self._planner_parameters,
                                            objectives=self.objectives,
                                            constraints=self.constraints,
                                            data=self.data,
                                            settings=self.settings,
                                            batch_size=request.batch_size)
                outcome= Outcome.SUCCESS
            except Exception as e:
                print(f' Planning Failed with error:{e}')
                plan_response = {n:{p:-1 for p in self.parameter_names} for n in range(request.batch_size)}
                outcome = Outcome.FAILURE

        ares_response = self._convert_plan_to_ares(plan_response) # Convert parameter set if necessary

        # NOTE: Inital condition overrides coming from ARES OS are given priority and ignore any relational constraints for implicit parameters
        for i, name in enumerate(self._ares_parameter_names):
            if name in initial_conditions:
                ares_response[0][name] = initial_conditions[name]
                initial_condition_override[i] = True

        # response.append((ares_response, outcome, initial_condition_override))

        return (ares_response, outcome, initial_condition_override)
    
    def _get_initial_conditions(self,request: PlanRequest) -> dict:
        response = dict()
        for p in request.parameters:
            if p.initial_value is not None:
                response[p.name] = p.initial_value
        return response


    def _convert_plan_to_ares(self,response:dict) ->dict:
        """Converts the planning respoinse from the the planner paramter set to the Ares Parameter Reponse

        Args:
            response (dict): Planner Reponse

        Returns:
            dict: Response with ARES Parameters
        """
        
        if self._ares_parameter_names == self._planner_parameter_names:
            ares_response = response
        else:
            ares_response = dict()
            for n in response.keys():
                batch_item = response[n]
                batch_response = dict()
            
                for p in self._ares_parameter_names:
                    if p in self._planner_parameter_names:
                        batch_response[p] = batch_item[p]
                    else:
                        batch_response[p] = self._eval_implicit(batch_item,p)

                ares_response[n] = batch_response
                
        return ares_response

    def _ax_log_interceptor(self):
        ax_logger = logging.getLogger("ax")
        ax_logger.handlers.clear()
        ax_logger.propagate = False
        log_stream = io.StringIO()
        handler = logging.StreamHandler(log_stream)
        ax_logger.addHandler(handler)
        ax_logger.setLevel(logging.INFO) # Set to DEBUG if you want more verbose output
    
        return log_stream, handler, ax_logger

    @property 
    def parameter_names(self) -> list[str]:
        return list([p['name'] for p in self.parameters])

    @property
    def objective_names(self) -> list[str]:
        return list(self.objectives.keys())
    @property 
    def _ares_parameter_names(self) -> list[str]:
        return list([p['name'] for p in self._ares_parameters])
    
    @property 
    def _planner_parameter_names(self) -> list[str]:
        return list([p['name'] for p in self._planner_parameters])
    @property 
    def _implicit_parameter_names(self) -> list[str]:
        return list([p['name'] for p in self._implicit_parameters])
    
    @property
    def data(self) -> list[dict]:
        # Downselects the paramters to pass to the planner and the data (seed data + previous trials) in a format compatible with the Ax API
        if len(self._seed_data_df) >0:
            working_seed_df = self._seed_data_df.copy()
        else:
            working_seed_df = pd.DataFrame(columns=self._planner_parameter_names+list(self.objectives.keys()))
        if self.settings['Parameter Value Type'] == "Planned":
            working_data_df = self._planned_df.copy()
        elif self.settings['Parameter Value Type'] == "Acheived":
            working_data_df = self._acheived_df.copy()
        else:
            raise Exception('Parameter Value Type must be Planned or Acheived')
        if len(working_data_df) == 0:
            working_data_df = pd.DataFrame(columns=self._planner_parameter_names+list(self.objectives.keys()))
        # Downselect both dataframes to just the planning paramters
        working_data_df = working_data_df[self._planner_parameter_names+list(self.objectives.keys())]
        working_seed_df = working_seed_df[self._planner_parameter_names+list(self.objectives.keys())]
        # Combine into a master df of what needs to be sent to the planner, then split into parameters and objective(s) dfs
        composite_df = pd.concat([working_seed_df,working_data_df],ignore_index=True).reset_index(drop=True)
        parameters_df = composite_df[self._planner_parameter_names]
        objectives_df = composite_df[list(self.objectives.keys())]
        
        p_dict = parameters_df.to_dict(orient='records')
        o_dict = objectives_df.to_dict(orient='records')
        data = []
        for i in range(len(p_dict)):
            data.append({'parameters':p_dict[i],
                         'objectives':o_dict[i]})
        return data
    @property
    def data_df(self)-> pd.DataFrame:
        data = self.data
        return pd.DataFrame([i['parameters']|i['objectives'] for i in data])
    
def plan_function(parameters: list[dict],
                  objective: dict,
                  constraints: list[str],
                  data:list[dict],
                  settings:dict) -> dict:
    
    # Dummy planner that just returns ones for all parameters
    p_names = [ p['name'] for p in parameters]
    
    return {p:1.0 for p in p_names}






