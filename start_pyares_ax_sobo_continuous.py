
from PyAres import AresPlannerService, AresDataType
from pyares_bo import sobo_cont_planner

if __name__ == "__main__":
  name = "PyARES SOBO Planner"
  description = "Single objective baysian optimization planner using Ax and Honegumi for continuous variables"
  version = "0.0.1"
  planner = AresPlannerService(sobo_cont_planner,
                               name,
                               description,
                               version, port=8002)

  #Mark that the planner supports numbers
  planner.add_supported_type(AresDataType.NUMBER)

  planner.add_setting('Seed Data', AresDataType.STRUCT)
  planner.add_setting('Minimize',AresDataType.BOOLEAN)
  planner.add_setting('Constraints',AresDataType.STRING_ARRAY)
  planner.add_setting("Verbose Output", AresDataType.BOOLEAN) #If enabled, will print out more detailed information during planning
  planner.add_setting("RNG Seed", AresDataType.NUMBER,optional=True) # Sets a seed for the random number generator
  planner.start()

