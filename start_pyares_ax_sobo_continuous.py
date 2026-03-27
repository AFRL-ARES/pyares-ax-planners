
from PyAres import AresPlannerService, AresDataType
from src.pyares_bo.ax import sobo_cont_planner

if __name__ == "__main__":
  name = "PyARES SOBO Planner"
  description = "Single objective baysian optimization planner using Ax and Honegumi for continuous variables"
  version = "0.0.1"
  planner = AresPlannerService(sobo_cont_planner,
                               name,
                               description,
                               version, port=1337)

  #Mark that the planner supports numbers
  planner.add_supported_type(AresDataType.NUMBER)

  # Path to seed data file. File should be formatted with column headers matching the parameter names coming from ARES OS
  planner.add_setting('Seed Data', AresDataType.STRING) 
  planner.add_setting('Minimize',AresDataType.BOOLEAN) # True: mininimze objective score, false: maximize objective score
  planner.add_setting('Constraints',AresDataType.STRING_ARRAY) # An array of strings that define constraints between parameters that will be passed to the underlying Ax API
  planner.add_setting("Verbose Output", AresDataType.BOOLEAN) #If enabled, will print out more detailed information during planning
  planner.add_setting("RNG Seed", AresDataType.NUMBER,optional=True) # Sets a seed for the random number generator
  planner.add_setting("Smart Seed",AresDataType.BOOLEAN) # If True and there is no seed data provided will run specified numbner of seed experiments before planing
  planner.add_setting("Smart Seed Type",AresDataType.STRING,constraints=['Latin Hyper Cube']) # The method used to generate the seed points
  planner.add_setting("Smart Seed Points",AresDataType.NUMBER) # How many seed experiments to do before starting BO
  planner.start()

