from PyAres import PlanRequest, PlanResponse, AresDataType, Outcome
from pathlib import Path
import pandas as pd
import warnings
from datetime import datetime
from ax.service.ax_client import ObjectiveProperties
import logging 
from ax.utils.common.logger import ROOT_STREAM_HANDLER
from typing import Any
import sympy as sp
ROOT_STREAM_HANDLER.setLevel(logging.WARNING) # Supresses Ax INFO messages

class PyAres_Ax_Planner(object):
    def __init__(self):
        self.settings : dict = {}
        self.settings_list = []
        self.add_setting(setting_name='Seed Data',setting_type=AresDataType.STRING)
        self.add_setting(setting_name='Constraints',setting_type=AresDataType.STRING_ARRAY)
        self.add_setting(setting_name='Implicit Parameters',setting_type=AresDataType.STRING_ARRAY) 
        self.add_setting(setting_name='Verbose Output',setting_type=AresDataType.BOOLEAN,default_value=True)
        # Note, it looks like acheived value isn't coming over need to make a github issue
        self.add_setting(setting_name='Parameter Value Type',setting_type=AresDataType.STRING,optional=False,constraints=['Planned','Acheived'],default_value='Planned')
        
        self.name : str = 'PyAres Prototype Ax Planner'
        self.version_number : str = 'x.x.x'
        self.description : str = "Parent class for making Ax API based planners compatible with ARES OS though PyARES"
        self.plan_function = plan_function
        self.planner_input_type = AresDataType.NUMBER
        self.acheived_values = {}
        self.planned_values = {}
        self.seed_data = []
        self.parameters = []
        self.implicit_parameters = []
        self.derived_values = []

    def info(self):
        return {'name':self.name,'version':self.version_number,'description':self.description}

    def configure_settings(self, planner):
        '''
        Returns a list of the planner settings that the planner start service can use to configure the planner
        Also initializes default settings, as that is not currently suported by PyAres
        '''
        planner.add_supported_type(self.planner_input_type)
        # TODO: Re-evaluate this once ARES OS and PyARES support communication of default values
        for s in self.settings_list:
            self.settings[s['setting_name']] = s['default_value']
            _s = dict(s)
            _ = _s.pop('default_value')# TODO: Reevaluate once defualt values are figured out on the ARES OS/PyARES side
            planner.add_setting(_s.pop('setting_name'), _s.pop('setting_type'),**_s)
        return planner
    def _configure_objectives(self):
        # Default _configure_objectives function. This will generally be overriden by a specific planner implementations
        self.objectives = {'objective':ObjectiveProperties(minimize=False)} # TODO: Update once support for multiple objectives is supported by ARES OS

    def _configure_parameters(self,request):
        # Implementing this as a function in case some planner implementaitons need to override it
        # TODO: Reevaluate logic down the road if flagging parameters as implicit is supported as part of ARES planning requests.
        '''
        In some cases parameters requested by ARES OS may be implicit or planning may need do occur based on values that are 
        derived from other paramters or the combinations thereof. This can be useful for summation constraints 
        (e.g., the total flow rate through the reactor must remain constant, Fractional compostion of all components must sum to 1) 
        or when planning must take place over a derived variable, (e.g., log([products]/log([reactants]), planning occurs over mole ractions but ARES OS
        provides total flows)
        Due to how Ax processes constraint strings a constraint such as x1 + x2 + x3 == total instead needs to be treated as 
        x1 + x2 <= total, with only x1 and x2 supplied to the planner and x3 = total - (x1 +x2)
        
        Implicit Paramters are defined through the planner settings though sympy compatible strings much in the way that constraints are
        Strings should have the form "<implicit parameter> = <expression>", and will be split and evaluated using the sympy library to 
        create a function that returns the value of the implicit parameter
        '''
        p_names = [p.name for p in request.parameters]
        if self.settings['Implicit Values'] == 0:
            self.parameters = [{'name':p.name, 
                                'type':'range',  # TODO: Update this once ARES OS requests explicity support categorical variables
                                'bounds':[p.minimum_value, p.maximum_value]} for p in request.parameters]
        else:
            is_implicit = []
            for value_str in self.settings['Implicit Values']:
                # Split out the paramter strings
                expr_elements = value_str.split('=')
                lhs = expr_elements[0].strip
                rhs = expr_elements[-1].strip
                # evaluate RHS to sympy expression then turn it into a function
                expr = sp.sympify(rhs) #TODO: Check on sympy input sanitization
                symbols = list(expr.free_symbols)
                symbol_names = [symbol.name for symbol in symbols]
                func = sp.lambdify(symbols, expr, modules='numpy')
                # Make a record of what the value name is, the evaluation function and what inputs (in order) go into it.
                if lhs in p_names:
                    self.implicit_parameters.append({'name':lhs, 'function':func, 'inputs':symbol_names})
                    is_implicit.append(lhs)
                else:
                    self.derived_values.append({'name':lhs, 'function':func, 'inputs':symbol_names})
            for p  in request.parameters:
                if p.name not in is_implicit:
                    self.parameters.append({'name':p.name,'type':'range','bounds':[p.minimum_value, p.maximum_value]})
                
        # TODO: Figure out how to handle telling the planner to plan over derived variables rather than the parameters recieved from ARES OS
                

    def call_planner(self,request: PlanRequest):
        start = datetime.now()
        self.N_trial = len(request.analysis_results)
        print(f"--- Planning Trial #{self.N_trial} ---")
        print(f' Planning Started at: {start.strftime("%Y-%m-%d %H:%M:%S")}')
    
        self.verbose = self.settings['Verbose Output']
        # Updates and sttings from their default value
        self.settings.update(request.settings)
        self.constraints = self.settings['Constraints']
        self._configure_parameters(request)
        self._configure_objectives()
        # Parsing historical data
        self._process_seed_data()
        self._process_experimental_data(request)

        # Overrides for initial values on first iteraton
        initial_condition_override = [False for i in self.parameter_names]
        if self.N_trial == 0:
            initial_conditions = self._get_initial_conditions(request)
            # If all parameters are present in the initial conditions response, skip planning
            # Planners could be aware of their own history so we don't want to call the planner unecessarily.
            if all(p in initial_conditions.keys() for p in self.parameter_names):
                plan_response = initial_conditions
                outcome = Outcome.SUCCESS

            else: 
            # If some values do not have an initial value specified we still need to get values from the planner
            # and overwrite the ones that we have initial values for.
                try:
                    plan_response = self.plan_function(parameters=self.parameters,
                                                objective=self.objectives,
                                                constraints=self.constraints,
                                                data=self.seed_data+self.data,
                                                settings=self.settings)
                    outcome= Outcome.SUCCESS
                except Exception as e:
                    print(f' Planning Failed with error:{e}')
                    plan_response = {p:-1 for p in self.parameter_names}
                    outcome = Outcome.FAILURE

                for i, name in enumerate(self.parameter_names):
                    if name in plan_response:
                        plan_response[name] = initial_conditions['name']
                        initial_condition_override[i] = True
        else:
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

            print(" Proposed test condition:")
            for n,v in zip(plan_response.keys(), plan_response.values()):
                print(f"\t{n} = {v:.3f}")
            end = datetime.now()
            delta= end-start
            print(f"--- End Planning for Trial #{self.N_trial} (Took {delta.total_seconds()}) seconds---")

        response = PlanResponse(parameter_names=list(plan_response.keys()),
                                parameter_values=list(plan_response.values()),
                                planning_outcome=outcome)
        return response

    def _process_experimental_data(self, request: PlanRequest):
            self.objective_values = request.analysis_results
            for p in request.parameters:
                self.acheived_values[p.name] = list([p.param_history[i].achieved_value for i in range(len(request.analysis_results))])
                self.planned_values[p.name] = list([p.param_history[i].planned_value for i in range(len(request.analysis_results))])

    def _process_seed_data(self):
        seed_as_path = Path(self.settings['Seed Data'])
        # for now we'll only deal with reading seed data from a file
        # Note that the objective value score in the seed data file must have the column name 'objective' (case sensitive)
        # TODO: Reevaluate once there are more data curation tools availible to ARES OS
        if self.settings['Seed Data'] != '':
            if seed_as_path.is_file():
                self.seed_data = self._process_seed_data_file(seed_as_path)
            else:
                warnings.warn(f"Seed data setting {self.settings['Seed Data']} is not recognized as a file. Data will not be used")
                self.seed_data = []
        else:
            self.seed_data = []

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

    def _process_seed_data_file(self,file:Path):
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
    
    def _get_initial_conditions(self,request: PlanRequest) -> dict:
        response = dict()
        for p in request.parameters:
            if p.initial_value is not None:
                response[p.name] = p.initial_value
        return response

    def add_setting(self, setting_name:str, 
                    setting_type:AresDataType,
                    optional: bool = True,
                    constraints: list = [],
                    default_value:Any= ''):
        
        self.settings_list.append({'setting_name':setting_name, 
                                   'setting_type':setting_type,
                                   'optional':optional,
                                   'constraints':constraints,
                                   'default_value':default_value})
    @property 
    def parameter_names(self) -> list[str]:
        return list([p['name'] for p in self.parameters])
    @property
    def data(self) -> list[dict]:
        data = list()
        # Return a list of dicts format compaible with giving values to the Ax api
        for i in range(len(self.objective_values)):
            if self.settings['Parameter Value Type'] == "Planned":
                par_dict = {key:self.planned_values[key][i] for key in self.planned_values}
            elif self.settings['Parameter Value Type'] == "Acheived":
                par_dict = {key:self.acheived_values[key][i] for key in self.acheived_values}
            else:
                raise Exception('Parameter Value Type must be Planned or Acheived')

            obj_dict = {'objective':self.objective_values[i]}
            data.append({'parameters':par_dict,
                        'objectives':obj_dict})
        return data
    
    @property
    def _planner_parameters(self) ->list[dict]:
        # TODO: Figure out support for derived parameters for planning
        return self.parameters

def plan_function(parameters: list[dict],
                  objective: dict,
                  constraints: list[str],
                  data:list[dict],
                  settings:dict) -> dict:
    
    # Dummy planner that just returns ones for all parameters
    p_names = [ p['name'] for p in parameters]
    
    return {p:1.0 for p in p_names}






