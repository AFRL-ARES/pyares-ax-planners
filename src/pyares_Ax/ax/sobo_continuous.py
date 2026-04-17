from .ax_compatibility import PyAres_Ax_Planner
from ax.service.ax_client import AxClient, ObjectiveProperties
from PyAres import AresDataType
from time import time

class SOBO_Ax_Planner(PyAres_Ax_Planner):
    def __init__(self):
        super().__init__()
        self.name = "SOBO Ax Planner"
        self.description = "Single Objective Bayesian Optimization planner for continuous variables using Ax"
        self.version_number = "0.1.0"
        self.plan_function = sobo_planner
        self.add_setting('Minimize', AresDataType.BOOLEAN,False)
        self.add_setting("RNG Seed", AresDataType.NUMBER,optional=True) # Sets a seed for the random number generator

    def _configure_objectives(self):
        # Override the parent class's objective setter function so we can use the planner specific behavior
        self.objectives = {'objective':ObjectiveProperties(minimize=self.settings['Minimize'])}

def sobo_planner(parameters:list[dict], 
                 objective:dict, 
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
        df = ax_client.get_trials_data_frame()
        # This is a bit Hacked in at the moment, need to get some stuff worked out with the metadata handling but this at least gets the data out
        folder = settings['_exp_output_dir']
        df.to_excel(str(folder/'campaign_progress.xlsx',))
    parameterization, _ = ax_client.get_next_trial()

    return parameterization
