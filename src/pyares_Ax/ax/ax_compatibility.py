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

from PyAres import PlanRequest, PlanResponse, AresDataType, Outcome, AresPlannerService
from pathlib import Path
import pandas as pd
import warnings
from datetime import datetime
from ax.service.ax_client import ObjectiveProperties
import logging 
from ax.utils.common.logger import ROOT_STREAM_HANDLER
from typing import Any, Callable
import sympy as sp
ROOT_STREAM_HANDLER.setLevel(logging.WARNING) # Supresses Ax INFO messages

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
        # Place Holders for for planner specific information, these should be overridden by child classes
        self.plan_function : Callable = plan_function
        self.planner_input_type : AresDataType = AresDataType.NUMBER

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
        self.add_setting(setting_name='Seed Data',setting_type=AresDataType.STRING)
        self.add_setting(setting_name='Constraints',setting_type=AresDataType.STRING_ARRAY)
        self.add_setting(setting_name='Implicit Values',setting_type=AresDataType.STRING_ARRAY) 
        self.add_setting(setting_name='Verbose Output',setting_type=AresDataType.BOOLEAN,default_value=True)
        self.add_setting(setting_name='Parameter Value Type',setting_type=AresDataType.STRING,optional=False,constraints=['Planned','Acheived'],default_value='Planned')
    
    ### Interface functions ###
    # These functions are expected to be implemented in all planner classes and are the primary way PyAres interfaces with the planner
    def info(self) -> dict:
        """
        Returns basic information about the planner that will be repored to ARES OS.
        Storing it in the planner class makes it easier to keep a single source of truth for these values
        """
        return {'name':self.name,'version':self.version_number,'description':self.description}
    
    def add_setting(self, setting_name:str, 
                    setting_type:AresDataType,
                    optional: bool = True,
                    constraints: list = [],
                    default_value:Any= ''):
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
                                   'default_value':default_value})
    
    def configure_settings(self, planner:AresPlannerService) -> AresPlannerService:
        '''
        Applies the planner settings to the AresPlannerService object. Also initializes default settings within the
        planner object's own internal settings atrribute.
        Note: 
        This initializes the internal settings atribute with default values so that any check of those settings 
        do not have to account for the possibly of a missing value. This behavior may change when PyAres messages
        are updated to support default values.
        '''
        # TODO: Re-evaluate this once ARES OS and PyARES support communication of default values
        planner.add_supported_type(self.planner_input_type)
        for s in self.settings_list:
            self.settings[s['setting_name']] = s['default_value']
            _s = dict(s)
            _ = _s.pop('default_value')
            planner.add_setting(_s.pop('setting_name'), _s.pop('setting_type'),**_s)
        return planner
    
    def call_planner(self,request: PlanRequest) -> PlanResponse:
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
        # Start Planning, start tracking time
        start = datetime.now()
        self.N_trial = len(request.analysis_results)
        print(f"--- Planning Trial #{self.N_trial} ---")
        print(f' Planning Started at: {start.strftime("%Y-%m-%d %H:%M:%S")}')

        # Updates settings from the request and parses a few essential values
        self.settings.update(request.settings)
        self.verbose = self.settings['Verbose Output']
        self.constraints = self.settings['Constraints']

        # Configure Planning parameters and objectives
        self._configure_parameters(request)
        self._configure_objectives()

        # Parse seed data and previous trials into a format that is useful for the planner
        self._process_seed_data()
        self._process_experimental_data(request)

        if self.N_trial == 0:
            # Separate logic to handle the first run of the planner for things specified initial conditions
            ares_response, outcome, override_flags = self._plan_first_run(request)
        else:
            override_flags = [False for i in self._ares_parameter_names]
            try:
                plan_response = self.plan_function(parameters=self._planner_parameters,
                                                objective=self.objectives,
                                                constraints=self.constraints,
                                                data=self.seed_data+self.data,
                                                settings=self.settings)
                outcome= Outcome.SUCCESS
            except Exception as e:
                print(f' Planning Failed with error:{e}')
                plan_response = {p:-1 for p in self.parameter_names}
                outcome = Outcome.FAILURE

            ares_response = self._convert_plan_to_ares(plan_response)

        print("Proposed test condition:")
        for i,(n,v) in enumerate(zip(ares_response.keys(), ares_response.values())):
            if override_flags[i]:
                 print(f"\t{n} = {v:.3f} (Overriden by supplied inital value))")
            else:
                print(f"\t{n} = {v:.3f}")

        end = datetime.now()
        delta= end-start
        print(f"--- End Planning for Trial #{self.N_trial} (Took {delta.total_seconds()}) seconds---")
        
        response = PlanResponse(parameter_names=list(ares_response.keys()),
                                parameter_values=list(ares_response.values()),
                                planning_outcome=outcome)
        return response
    
    ### Support functions - Not intended for general interfacing
    ##Override these functions when configuring your planner subclass
    def _configure_objectives(self):
        """
        Configures the objectives for the Ax planner. This function is expected to be overridden in the child class to propperly set the planning goals.

        Note: Due to the current structure of the PlanRequest message, PyAres only officially supports single-objective planning at the moment
        this constriaint should be relaxed in a future release
        """
        # TODO: Update once support for multiple objectives is better supported by ARES OS/PyAres
        self.objectives = {'objective':ObjectiveProperties(minimize=False)} 

    ## General support functions, may be overridden if necessary
    def _configure_parameters(self,request):
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
        
        if len(self.settings['Implicit Values']) == 0 and self.settings['Planning Parameter Override'] == '':
            self._planner_parameters = self._ares_parameters
        else:
            for value_str in self.settings['Implicit Values']:
                # Split out the paramter strings
                expr_elements = value_str.split('=')
                lhs = expr_elements[0].strip
                rhs = expr_elements[-1].strip
                # evaluate RHS to sympy expression then turn it into a function
                expr = sp.sympify(rhs) 
                symbols = list(expr.free_symbols)
                symbol_names = [symbol.name for symbol in symbols]
                func = sp.lambdify(symbols, expr, modules='numpy')
                # Make a record of what the value name is, the evaluation function and what inputs (in order) go into it.
                value_dict = {'name':lhs, 'function':func, 'inputs':symbol_names,'expression':value_str}
                # If the parameter is an ares parameter it gets stored in self._implicit_parameters, otherwise it gets stored in self._implicit_values
                if lhs in self._ares_parameter_names: 
                    self.implicit_parameters.append(value_dict)
                else:
                    self._implicit_values.append(value_dict)
                #sets the values that the planner will actuall operate on
                self._planner_parameters = [p for p in self._ares_parameters if p.name not in self._implicit_parameter_names]                
        # TODO: Figure out how to handle telling the planner to plan over derived variables rather than the parameters recieved from ARES OS and how to solve 
        # the degeneracy and bounds issues that come with that capibility

    def _process_experimental_data(self, request: PlanRequest):
        """
        Processes the experimental trial data in the PyAres request into a format that the planner can use.

        Args:
            request (PlanRequest): PyAres Planning request
        """
        self.objective_values = request.analysis_results
        for p in request.parameters:
            self.achieved_values[p.name] = list([p.param_history[i].achieved_value for i in range(len(request.analysis_results))])
            self.planned_values[p.name] = list([p.param_history[i].planned_value for i in range(len(request.analysis_results))])

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
                self.seed_data = self._process_seed_data_file(seed_as_path)
            else:
                warnings.warn(f"Seed data setting {self.settings['Seed Data']} is not recognized as a file. Data will not be used")
                self.seed_data = []
        else:
            self.seed_data = []

    def _process_seed_data_file(self,file:Path) ->list[dict]:
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
        
        # Format the data in to list of dicts format that the Ax planner expects
        data = list()
        for i in range(len(df)):
            obj_dict = {'objective':df['objective'][i]}
            par_dict = {p:df[p][i] for p in self.parameter_names}
            data.append({'parameters':par_dict,
                        'objectives':obj_dict})
        return data
    def _eval_implicit(self,input_dict:dict, target_name:str):
        """
        Calculates the target implicit value or factor based on inputs and definitions.
        Args:
            input_values (dict): A dictionary of explicit inputs (e.g., {'x1': 5, 'x2': 10}).
            target_name (str): The name of the parameter/factor to compute.
            
        Returns:
            The evaluated result for the target_name.
        """
        implicit_params = self.implicit_parameters
        derived_vals = self.derived_values

        
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
                                            objective=self.objectives,
                                            constraints=self.constraints,
                                            data=self.seed_data+self.data,
                                            settings=self.settings)
                outcome= Outcome.SUCCESS
            except Exception as e:
                print(f' Planning Failed with error:{e}')
                plan_response = {p:-1 for p in self._planner_parameter_names}
                outcome = Outcome.FAILURE

        ares_response = self._convert_plan_to_ares(plan_response) # Convert parameter set if necessary

        # NOTE: Inital condition overrides coming from ARES OS are given priority and ignore any relational constraints for implicit parameters
        for i, name in enumerate(self._ares_parameter_names):
            if name in ares_response:
                ares_response[name] = initial_conditions['name']
                initial_condition_override[i] = True
        
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
        ares_response = dict()
        if self._ares_parameter_names == self._planner_parameter_names:
            ares_reponse = response
        else:
            for p in self._ares_parameter_names:
                if p in self._planner_parameter_names:
                    ares_response[p] = response[p]
                else:
                    ares_response[p] = self._eval_implicit(response,p)
        
        return ares_response

    @property 
    def parameter_names(self) -> list[str]:
        return list([p['name'] for p in self.parameters])
    
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
        data = list()
        # Return a list of dicts format compaible with giving values to the Ax api
        for i in range(len(self.objective_values)):
            if self.settings['Parameter Value Type'] == "Planned":
                par_dict = {key:self.planned_values[key][i] for key in self.planned_values}
            elif self.settings['Parameter Value Type'] == "Acheived":
                par_dict = {key:self.achieved_values[key][i] for key in self.achieved_values}
            else:
                raise Exception('Parameter Value Type must be Planned or Acheived')

            obj_dict = {'objective':self.objective_values[i]}
            data.append({'parameters':par_dict,
                        'objectives':obj_dict})
        return data
    
def plan_function(parameters: list[dict],
                  objective: dict,
                  constraints: list[str],
                  data:list[dict],
                  settings:dict) -> dict:
    
    # Dummy planner that just returns ones for all parameters
    p_names = [ p['name'] for p in parameters]
    
    return {p:1.0 for p in p_names}






