from ax.service.ax_client import AxClient
from ax.service.utils.instantiation import ObjectiveProperties
import numpy as np

params = [{'name':'A','bounds':[-1.0,1.0],'type':'range'},{'name':'B','bounds':[-10.0,10.0],'type':'range'}]
N=3
A = np.random.uniform(-1,1,N)
B = np.random.uniform(-10,10,N)
O = np.random.normal(5,1,N)

ax_client = AxClient()
objective = {'objective':ObjectiveProperties(minimize=True)}
ax_client.create_experiment(parameters=params,
                            objectives=objective,)

# ax_client.configure_generation_strategy(initialization_budget=5,initialize_with_center=False,use_existing_trials_for_initialization=True)
for i,o in enumerate(O):
    _,ti = ax_client.attach_trial(parameters={"A":A[i],"B":B[i]})
    ax_client.complete_trial(trial_index=ti,raw_data={'objective':O[i]})

for i in range(10):
    p,ti = ax_client.get_next_trial()
    ax_client.complete_trial(trial_index=ti,raw_data={'objective':np.random.normal(5,1)})

