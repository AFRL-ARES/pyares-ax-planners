from .ax_compatibility import PyAres_Ax_Planner
from ax.service.ax_client import AxClient
from ax.service.utils.instantiation import ObjectiveProperties
from PyAres import AresDataType, PlanRequest
from time import time
from pathlib import Path
import pandas as pd

class SOBO_Ax_Planner(PyAres_Ax_Planner):
    def __init__(self):
        super().__init__()
        self.name = "SOBO Ax Planner"
        self.description = "Single Objective Bayesian Optimization planner for continuous variables using Ax"
        self.version_number = "0.1.0"
        self.plan_function = sobo_planner
        self.add_setting('Minimize', AresDataType.BOOLEAN,False)
        self.add_setting("RNG Seed", AresDataType.NUMBER,optional=True) # Sets a seed for the random number generator
        # This is a hack to make things work with the current limitations of Athena. By pointing both the planner and
        # analyzer to the same file we can circumvent the current data handling limitations of ARES OS v2.1.0
        self.add_setting("Swap File", AresDataType.STRING) 

    def _configure_objectives(self):
        # Override the parent class's objective setter function so we can use the planner specific behavior
        self.objectives = {'objective':ObjectiveProperties(minimize=self.settings['Minimize'])}
    
    # We need to override the data intake logic so that the planner switches to using the swap file after
    def _process_seed_data(self):
        # If the swap file exists it will already incorporate the data from the swap file so we can just read one file
        swap_file = self.settings.get('Swap File','')
        if swap_file != '' and Path(swap_file).exists() and Path(swap_file).is_file():
            self._seed_data_df = self._process_seed_data_file(Path(swap_file))
        else:
            super()._process_seed_data()

    def _process_experimental_data(self, request: PlanRequest):
        # We need to override the experimental data processing if we're using a swap file.
        # The swap file should already be updated with the most recent results, so we don't 
        # need any of the experimental info in the request file.
        swap_file = self.settings.get('Swap File','')
        if swap_file != '' and Path(swap_file).exists() and Path(swap_file).is_file():
            cols = self._ares_parameter_names + list(self.objectives.keys())
            planned_df = pd.DataFrame(columns=cols)
            achieved_df = pd.DataFrame(columns=cols)
            self._acheived_df = achieved_df
            self._planned_df = planned_df
        else:
            super()._process_experimental_data(request)      
    
    def cleanup(self):
        #TODO: Update this when Metadata handling is fixed on the ARES OS side
        campagin_id = self._latest_metadata.get('campaign_id','')
        swap_file = self.settings.get('Swap File','')
        if swap_file != '' and Path(swap_file).exists() and Path(swap_file).is_file():
            file = Path(swap_file)
            file = file.resolve()
            name = file.parent/(file.stem+'_camp_id_'+campagin_id+file.suffix)
            file.rename(str(name))



            



def sobo_planner(parameters:list[dict], 
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
        # This is a bit Hacked in at the moment, need to get some stuff worked out with the metadata handling but this at least gets the data out
        folder = settings['_exp_output_dir']
        df.to_csv(str(folder/'campaign_progress.xlsx'),index=False)
    parameterization, _ = ax_client.get_next_trial()
    swap_df = ax_client.get_trials_data_frame()
    make_swap_file(swap_df,settings,parameters,objectives)

    return parameterization

def make_swap_file(df,settings,parameters,objectives):
    swap_file = settings.get('Swap File','')
    if swap_file != '':
        swap_file = Path(swap_file)
        swap_file.parent.mkdir(exist_ok=True,parents=True)
        p_names = [p.get('name') for p in parameters]
        o_names = list(objectives.keys())
        if len(df) == 1:
            sdf = df[p_names]
            sdf[o_names] = ''
        else:
            sdf = df[p_names+o_names]
        sdf.to_csv(str(swap_file),index=False)




