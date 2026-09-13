from .ax_compatibility import PyAres_Ax_Planner
from ax.service.ax_client import AxClient
from ax.service.utils.instantiation import ObjectiveProperties
from PyAres import AresDataType, PlanRequest
from time import time
from PyAres.Models import AresSchemaEntry

class MOBO_Ax_Planner(PyAres_Ax_Planner):
    def __init__(self):
        super().__init__()
        self.name = "MOBO Ax Planner"
        self.description = "Multi Objective Bayesian Optimization planner for continuous variables using Ax"
        self.version_number = "0.1.0"
        self.plan_function = mobo_planner # The Function to call for planning

        # Allows the user to specific the names of objectives to minimize in the service settings. 
        minimize_setting_schema = AresSchemaEntry(type=AresDataType.STRUCT,
                                                  struct_schema={"objective_name":AresSchemaEntry(type=AresDataType.STRING,
                                                                                                  description='Name of the objective to apply the setting to. Must match exactly.'),
                                                                 "minimize":AresSchemaEntry(type=AresDataType.BOOLEAN,
                                                                                            description='Whether the objective should be minimized.')
                                                                }
                                                )
        self.add_setting(setting_name = 'Minimize',
                         setting_type = AresDataType.LIST,
                         list_element_schema = minimize_setting_schema,
                         description= "Configure whether objectives should be minimized or maximized. Default Behavior is to Maximize")

        self.add_setting(setting_name="RNG Seed", 
                         setting_type=AresDataType.INT,
                         optional=True,
                         default_value=int(time()),
                         description="Random Number Generator Seed.") # Sets a seed for the random number generator

    def _configure_objectives(self,request:PlanRequest):
        # Override the parent class's objective setter function so we can use the planner specific behavior
        #TODO: Revisit once messaging protocol gets updated to include objective names before the first experiment
        if self.N_trial < 1: # We may not be getting the objective names on the first (pre-experiment) call to the planner
            self.objectives = {'objective':ObjectiveProperties(minimize=False)} 
        else:
            objective_names = [o.objective_name for o in request.analysis_objectives[0]]
            # TODO update settings to support multiple objectives with different goals
            objectives = {}
            minimization_setting = {item['objective_name']:item['minimize'] for item in request.settings['Minimize']}
            for name in objective_names:
                if name in minimization_setting.keys():
                    value = minimization_setting[name]
                else:
                    value = False
                objectives[name] =ObjectiveProperties(minimize=value)
            self.objectives = objectives

def mobo_planner(parameters:list[dict], 
                 objectives:dict, 
                 constraints:list[str], 
                 data:list[dict],
                 settings:dict) -> dict:
    """
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
    if not isinstance(settings['RNG Seed'],int):
        settings['RNG Seed'] = int(time())
    ax_client._random_seed = settings['RNG Seed']
    try:
        ax_client.create_experiment(parameters=parameters,
                                    objectives=objectives,
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
        df = ax_client.get_trials_data_frame()
        folder = settings['_exp_output_dir']
        df.to_excel(str(folder/'campaign_progress.xlsx',))
    parameterization, _ = ax_client.get_next_trial()

    return parameterization