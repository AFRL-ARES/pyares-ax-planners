from PyAres import AresPlannerService
from pyares_Ax.ax import MOBO_Ax_Planner
import argparse

if __name__ == "__main__":
    # Add command line argument parsing. Allows setting to me modified from ARES OS launcher PyAres tab
    parser = argparse.ArgumentParser(
        prog='PyAres_mobo_planner',
        description="Starts a Multi Objective Bayesian Optimization planner powered by Meta's Ax framework",
    )
    parser.add_argument('-p','--port',help='Port to host the planner service on',type=int, default=1336)
    parser.add_argument('-l','--local',help='Whether to run the service only on localhost', type=bool, default=True)
    parser.add_argument('-bp','--bokeh_port',help='Port for the Bokeh Visualizer Service', type=int, default=2336)
    args = parser.parse_args()
    port = args.port
    local = args.local
    bokeh_port = args.bokeh_port

    plan_object = MOBO_Ax_Planner()
    plan_object.visualizer_port = 2336
    planner_info = plan_object.info()
    plan_function= plan_object.call_planner

    planner= AresPlannerService(plan_function,
                                planner_info['name'],
                                planner_info['description'],
                                planner_info['version'],
                                port=port, use_localhost=local,multi_objective_capable=True)
    
    planner = plan_object.configure_settings(planner)
    planner.start()

