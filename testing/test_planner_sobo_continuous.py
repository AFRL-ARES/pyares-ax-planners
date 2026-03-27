import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from synthetic_process_space import SyntheticProcessResponse
from PyAres import PlanRequest, PlanningParameter, AresDataType, ParameterHistoryItem
from PyAres.test_tools import PlannerTestClient
import itertools

def run_test(test_client, 
                settings_dict, 
                params_dict, 
                response_surface,
                N_iterations,
                N_params):
    
    param_names = params_dict['names']
    if 'initial_values' in params_dict:
        init_vals = params_dict['initial_values']
    bounds = params_dict['bounds']

    response_dict = dict()
    # Run Planning Loop using the synthetic process response space
    results = []
    param_histories = [[] for i in range(N_params)]

    for i in range(N_iterations):
        planning_parameters = []
        if i > 0:
            results.append(response_surface.evaluate(response_dict))
        for j in range(len(param_names[:N_params])):
            if i == 0: # The ARES OS loop always starts with Plan, so the first entry will will have no result and no paramter history.
                param_histories[j].append(ParameterHistoryItem(planned_value=[], achieved_value=[]))
            else:
                val = response_dict[param_names[j]]
                param_histories[j].append(ParameterHistoryItem(planned_value=float(val), achieved_value=float(val))) 
            if 'initial_values' in params_dict:
                planning_parameters.append(PlanningParameter(name=param_names[j],
                                                            minimum_value=bounds[j][0],
                                                            maximum_value=bounds[j][1],
                                                            param_history=param_histories[j],
                                                            data_type=AresDataType.NUMBER,
                                                            is_planned=True,
                                                            is_result=False,
                                                            planner_name="Simulated Annealing Planner",
                                                            initial_value=init_vals[j,0]))
            else:
                planning_parameters.append(PlanningParameter(name=param_names[j],
                                                            minimum_value=bounds[j][0],
                                                            maximum_value=bounds[j][1],
                                                            param_history=param_histories[j],
                                                            data_type=AresDataType.NUMBER,
                                                            is_planned=True,
                                                            is_result=False,
                                                            planner_name="Simulated Annealing Planner"))

        request = PlanRequest(planning_parameters, settings_dict,results)
        response = test_client.run_planning(request)
        values = [v.number_value for v in response.parameter_values]
        response_dict = dict(zip(response.parameter_names, values))

    return planning_parameters, np.array(results)

def plot_params(n_iter, planning_parameters, results ):
    # 1. 2d plot of the variations in the parameters
    marker_cycle = itertools.cycle(('o','+','.','*','^','s','x','D'))

    fig, (ax_t,ax_b) = plt.subplots(2,1,sharex=True) # 2 plots showing the variation in the paameters (top) and the change in the objective function

    ax_t.set_xlim(-1, n_iter+1)
    ax_t.set_ylim(-0.05,1.05)
    ax_t.set_yticks([0,0.5,1])
    ax_t.set_yticklabels([r'$x_{min}$','',r'$x_{max}$'])
    ax_t.set_xlabel('Iteration',fontsize=12,fontweight='bold')
    ax_t.set_ylabel('Parameter Value',fontsize=12,fontweight='bold')

    for p in planning_parameters:
            name = p.name
            p_bounds = (p.minimum_value,p.maximum_value)
            values = []
            for iteration in p.param_history[1:]: # Drop the first entry as a it is empty
                values.append(iteration.planned_value)
            norm_values = (np.asarray(values) - p_bounds[0]) / (p_bounds[1]-p_bounds[0])
            ax_t.plot(np.arange(0,n_iter-1),norm_values,label=name,marker=next(marker_cycle))
            ax_t.legend(loc='center right')
    best_results = np.array([np.min(results[:i+1]) for i in range(len(results))])
    
    ax_b.set_ylim([np.min(results),np.max(results)])
    ax_b.set_yticks([np.min(results),np.max(results)])
    ax_b.set_ylabel('Objective Score (lower is better)',fontsize=12,fontweight='bold')
    ax_b.plot(np.arange(0,n_iter-1),results,label='Current Score',marker='o')
    ax_b.plot(np.arange(0,n_iter-1),best_results,label='Best Score',marker='D')
    ax_b.legend(loc='center right')
    fig.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Settings
    N_iterations = 50 # Number of "experiments" to run
    test_seed = 7654321098
    plan_seed = 1234567890 
    N_params = 4  # Number of parameters to optimize
    N_tests = 100 # Number of trials to run to gather average performance data

    # Define Test Data:
    settings_dict = {'Minimize':True,
                     "Verbose Output":True,
                     "RNG Seed":plan_seed}
    
    param_names = ["x1", 
                   "x2",
                   "x3",
                   "x4",
                   "x5"]

    rng = np.random.default_rng(seed=test_seed)

    bounds = [(0,100),
              (120,300),
              (-5,5),
              (0,1),
              (-10,10)]
    
    process_response = SyntheticProcessResponse(dict(zip(param_names[:N_params], bounds)),
                                                output_bounds=(0,1),
                                                num_gaussians=int(rng.integers(3,9)), 
                                                noise_scale=rng.uniform(0.05,0.2), 
                                                noise_frequency=rng.uniform(1.0,10.0), 
                                                seed=int(rng.integers(1e6,1e12)))
    
    params_dict ={'names':param_names,
                  'bounds':bounds}
    
    # Start Test Client
    test_client = PlannerTestClient(port=1337, host='localhost')

    # 1. Service Health Checks
    test_client.check_status()
    test_client.get_info()
    # 2. Run a test 
    planning_parameters, results = run_test(test_client,
                                            settings_dict,
                                            params_dict,process_response,
                                            N_iterations,
                                            N_params)
    
    plot_params(N_iterations,planning_parameters,results)
    