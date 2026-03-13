from ax.service.ax_client import AxClient, ObjectiveProperties
from PyAres import PlanRequest, PlanResponse


ax_client:AxClient | int = -1
parameters:list = []
iterations:int = 0
constraints:list =  []
trial_index: int = -1

def sobo_cont_planner_persist(request: PlanRequest) -> PlanResponse:
    # PyARES compatible single objective bayesian optimization planner based on the default Ax behavior
    # This version uses a persistent Ax experimental client
    global ax_client
    global parameters
    global iterations
    global constraints
    global trial_index

    verbose = request.settings['Verbose Output']
    init_client = False
    # Check to make sure that constraints and parameters match, that way the service doesn't need to be 
    # manually restarted if the user starts a new type of experiment

    if ax_client == -1:
        init_client = True
    req_contstraints = request.settings['Constraints']
    if req_contstraints != constraints:
        constraints = req_contstraints
        init_client = True

    req_parameters = [{'name':p.name,
                       'type':'range',
                       'bounds':[p.minimum_value, p.maximum_value]} for p in request.parameters]
    if req_parameters != parameters:
        parameters = req_parameters
        init_client = True
    
    if init_client:
        print("No Ax API client found or experimental configuration changed. Initializing...")
        ax_client = AxClient()
        iterations = 0
        objective = {'score':ObjectiveProperties(minimize=request.settings['Minimize'])}
        ax_client.create_experiment(parameters=parameters,
                                    objectives=objective,
                                    parameter_constraints=constraints) # type: ignore
    # TODO: Attach Initial Data

    # TODO: Parse Initial Values
    
    print(f'--- Planning Trial #{iterations} ---')
    if iterations == 0:
        # Plan First trial
        parameterization, trial_index = ax_client.get_next_trial()
    else:
        ax_client.complete_trial(trial_index=trial_index, raw_data=request.analysis_results[-1]) # type: ignore
        parameterization, trial_index = ax_client.get_next_trial()
    
    parameter_names = list(parameterization.keys())
    new_test_condition = list(parameterization.values())
    return PlanResponse(parameter_names=parameter_names, parameter_values=new_test_condition)


def sobo_cont_planner(request: PlanRequest) -> PlanResponse:
    # PyARES compatible single objective bayesian optimization planner based on the default Ax behavior
    # This version initilizes a new Ax API client for each request 
    constraints = request.settings['Constraints']
    ax_client = AxClient()
    parameters = [{'name':p.name,
                    'type':'range',
                    'bounds':[p.minimum_value, p.maximum_value]} for p in request.parameters]
    objective = {'score':ObjectiveProperties(minimize=request.settings['Minimize'])}
    ax_client.create_experiment(parameters=parameters,
                                objectives=objective,
                                parameter_constraints=constraints)

    # TODO: Attach Seed Data
    N_trials = len(request.analysis_results)
    print(f'--- Planning Trial #{N_trials} ---')
    if N_trials == 0:
        parameterization, _ = ax_client.get_next_trial()
        # If any of the parameters have a specified initial value overwrite the planner suggestion
        for p in request.parameters:
            if isinstance(p.initial_value, float):
                print(f'\tInitial value found: {p.name}: p.initial_value')
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
