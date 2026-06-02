from PyAres import AresPlannerService, AresDataType
# from src.pyares_Ax.ax import PyAres_Ax_Planner
from pyares_Ax.ax import SOBO_Athena as SOBO_Ax_Planner

if __name__ == "__main__":
    plan_object = SOBO_Ax_Planner()
    planner_info = plan_object.info()
    plan_function= plan_object.call_planner

    planner= AresPlannerService(plan_function,
                                planner_info['name'],
                                planner_info['description'],
                                planner_info['version'],
                                port=1337)
    
    planner = plan_object.configure_settings(planner)
    try:
        planner.start()
    except Exception as e:
        print(f"An Exception Occured {e}")
    finally:
        plan_object.cleanup()

