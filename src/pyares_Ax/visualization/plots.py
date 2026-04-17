import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PyAres import PlanRequest
from pathlib import Path
from datetime import datetime
import itertools
import numpy as np

"""
Some basic plotting funcationality, Once ARES OS has a propper visualiation service this will probably be replaced.
"""

def plot_trials_progress(request:PlanRequest):
    # Get all ths stuff we need for saving the output
    output_folder = request.settings['Output Folder']
    campaign_name = request.request_metadata.campaign_name
    n_iter = len(request.analysis_results)
    # TODO: possible issues with this if the day ticks over during a campaing. Can we send over the campaign start time as well?
    experiment_time = datetime.strptime(request.request_metadata.experiment_start_time, "%Y-%m-%d %H:%M:%S")
    experiment_date = experiment_time.strftime("%Y-%m-%d") # Just getting the YMD to put in the name for easy sorting
    experiment_time = experiment_time.strftime("%Y-%m-%dT%H-%M")

    experiment_name = f'{experiment_time}_experiment_{n_iter}'
    # experiment_id = request.request_metadata.experiment_id

    write_folder = Path(output_folder)/(experiment_date +'_'+campaign_name)/experiment_name
    write_folder.mkdir(exist_ok=True,parents=True)
    
    params = [{'name':p.name,
              'bounds':[p.minimum_value, p.maximum_value],
               'planned_values':np.array([p.param_history[i].planned_value for i in range(n_iter)]),
               'acheived_values':np.array([p.param_history[i].achieved_value for i in range(n_iter)])} for p in request.parameters]
    
    for p in params:
        p['norm_planned'] = (p['planned_values']-p['bounds'][0])/(p['bounds'][1]-p['bounds'][0]) 
        p['acheived_values'] = np.array([np.nan if item is None else item for item in p['acheived_values']])
        p['norm_acheived'] = (p['acheived_values']-p['bounds'][0])/(p['bounds'][1]-p['bounds'][0])

    obj_score = request.analysis_results
    best_score = []
    for i, s in enumerate(obj_score):
        if i == 0:
            best_score.append(s)
        else:
            if request.settings["Minimize"]:
                if s < best_score[-1]:
                    best_score.append(s)
                else:
                    best_score.append(best_score[-1])
            elif not request.settings["Minimize"]:
                if s > best_score[-1]:
                    best_score.append(s)
                else:
                    best_score.append(best_score[-1])

    best_score = np.array(best_score)
    obj_score = np.array(list(obj_score).copy())
    marker_cycle = itertools.cycle(('o','+','.','*','^','s','x','D')) 
    # Make the plot
    fig, (ax0, ax1) = plt.subplots(2,1,sharex=True,height_ratios=(1,1))
    # top plot: The planner score
    ax0.set_ylabel('Objective Score',fontsize=10,fontweight='bold')

    ax0.plot(np.arange(0,len(obj_score)),obj_score,label='Objective Score',marker='^')
    ax0.plot(np.arange(0,len(best_score)),best_score,label='Best Score',marker='s')
    ax0.legend(loc='center right',fontsize=9)

    # bottom plot: the changes in parameters
    if n_iter < 5: 
        ax1.set_xlim(-0.5,6)
    else:
        ax1.set_xlim(-0.5,n_iter+1)

    ax1.set_ylim(-0.1,1.1)
    ax1.set_yticks([0,0.5,1])
    ax1.set_yticklabels([r'$x_{min}$','',r'$x_{max}$'])
    ax1.set_xlabel('Iteration',fontsize=10,fontweight='bold')
    ax1.set_ylabel('Param. Value',fontsize=10,fontweight='bold')
    for j, p in enumerate(params):
            ax1.plot(np.arange(0,len(p['planned_values'])),p['norm_planned'],label=p['name'],marker=next(marker_cycle))
    ax1.legend(loc='center right',fontsize=9)
    fig.tight_layout()
    fig.savefig(str(write_folder/'params_objective_plot.png'), dpi=300)
    # plt.close('all')