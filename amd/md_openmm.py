#!/usr/bin/local/python3
"""
    MD simulation file for complete pembrolizumab. Use OpenMM.
    
    Usage
    =====
        python3 md_openmm.py -pdb <path>
"""

#  Import libraries.
from openmm.app import *
from openmm import *
from openmm.unit import *
import sys
from sys import stdout
import argparse

__author__ = "Ravy LEON FOUN LIN"
__date__ = "28/02/2024"

#  Retrieve input pdb file.
parser = argparse.ArgumentParser()
parser.add_argument("-pdb", help="Path to the PDB file.")
args = parser.parse_args()
PDB_FILE = args.pdb

#  Check if argument is set.

if PDB_FILE == None :
    parser.print_help()
    sys.exit()

def setup_system(PDB_FILE : str) :
    """This function define the system object use for simulation.

    Args:
        PDB_FILE (str): Path to the PDB file input.
    """
    
    print("Define GPU...")
    #  Specify to run on CUDA.
    platform = Platform.getPlatformByName('CUDA')
    
    print("Load PDB file...")
    #  Load PDB file input.
    pdb = PDBFile(PDB_FILE)
    
    print("Define ForceField...")
    #  Define forcefield.
    forcefield = ForceField("charmm36.xml", "charmm36/spce.xml")
    
    print("Define molecule...")
    #  Define molecule.
    modeller = Modeller(pdb.topology, pdb.positions)
    modeller.deleteWater()
    residues=modeller.addHydrogens(forcefield)
    
    print("Define system...")
    #  Define system.
    system = forcefield.createSystem(modeller.topology, nonbondedMethod=PME, nonbondedCutoff=0.8*nanometer, constraints=HBonds)
    
    print("Define integrators...")
    #  Define integrators.
    integrator = openmm.CompoundIntegrator()    #  CompoundIntegrator() allows the set of several integrators. Allowing the change during the different steps.
    
    print("Add integrators...")
    #  Add the integration methods.
    integrator.addIntegrator(LangevinMiddleIntegrator(300*kelvin, 1/picosecond, 0.002*picoseconds))
    integrator.addIntegrator(VerletIntegrator(0.001*picosecond))
    integrator.addIntegrator(AMDIntegrator(0.001*picosecond,10,-1600*kilojoule_per_mole))
    
    print("Define simulation system...")
    #  Define the simulation object.
    simulation = Simulation(modeller.topology, system, integrator, platform)
    simulation.context.setPositions(modeller.positions)
    
    return (simulation, integrator, system)

def minimization_step(SIMULATION) :
    """Perform minimization of the system.

    Args:
        SIMULATION (OpenMM object) : Simulation object define at the setup_system step. 
    """
    
    #  Retrieve energy level before energy minimization.
    energy_level = SIMULATION.context.getState(getEnergy=True).getPotentialEnergy()
    
    print(f"Energy level before the minimization step : {energy_level}")
    print("Minimization...")
    #  Minimization step.
    SIMULATION.minimizeEnergy()
    
    #  Retrieve energy level after minimization.
    energy_level = SIMULATION.context.getState(getEnergy=True).getPotentialEnergy()
    print(f"Energy level after minimization step : {energy_level}")
    
    return SIMULATION
    
def NVT_equilibration(SIMULATION, INTEGRATOR) :
    """Run NVT equilibration.

    Args:
        SIMULATION : Simulation object minimized.
        INTEGRATOR : Group of integrators.
    """
    
    #  Set the first integrator which allow to fix a temperature.
    print("Set integrator...")
    INTEGRATOR.setCurrentIntegrator(0)
    
    #  Set up reporters.
    print("Create reporters...")
    ## Retrieve structures.
    SIMULATION.reporters.append(PDBReporter('NVT.pdb', 10000))
    
    ## Print the data on terminal.
    SIMULATION.reporters.append(StateDataReporter(stdout, 1000, step=True,
            potentialEnergy=True, temperature=True, volume=True, time=True))
    
    ## Write metrics.
    SIMULATION.reporters.append(StateDataReporter("NVT_md_log.txt", 100, step=True,
            potentialEnergy=True, temperature=True, volume=True))
    
    ## Write trajectory.
    SIMULATION.reporters.append(DCDReporter("NVT.dcd",100))
    
    print("Running NVT...")
    SIMULATION.step(50000)
    
    return SIMULATION

def NPT_equilibration(SIMULATION, INTEGRATOR, SYSTEM) :
    """Run NPT equilibration.

    Args:
        SIMULATION : Simulation object minimized and NVT equilibrated.
        INTEGRATOR : Group of integrators.
    """
    
    #  Add barostat to keep pressure.
    print('Add barostat...')
    SYSTEM.addForce(MonteCarloBarostat(1*bar, 300*kelvin))
    
    #  Reinitialize the simulation to save the previous states.
    print("Reinitialize the simulation...")
    SIMULATION.context.reinitialize(preserveState=True)
    
    #  Set new reporters.
    print("Set new reporters NPT...")
    ##  New save of the structure.
    SIMULATION.reporters.append(PDBReporter('NPT.pdb', 10000))

    #  stdout output.
    SIMULATION.reporters.append(StateDataReporter(stdout, 1000, step=True,
        potentialEnergy=True, temperature=True, volume=True, density=True, time=True))
    
    #  Log file.
    SIMULATION.reporters.append(StateDataReporter("NPT_md_log.txt", 100, step=True,
        potentialEnergy=True, temperature=True, volume=True, density=True))
    
    #  Trajectory file.
    SIMULATION.reporters.append(DCDReporter("NPT.dcd",100))    
    
    #  Run NPT.
    print("Running NPT...")
    SIMULATION.step(50000)

    #  Reinitialize to save modifications.
    SIMULATION.context.reinitialize(preserveState=True)
    
    #  Write topology.
    PDBFile.writeFile(SIMULATION.topology, SYSTEM, open('equilibrate.pdb','w'))

    return (SIMULATION, SYSTEM)

def classic_md(SIMULATION,INTEGRATOR) :
    """Run a classic MD using a leap-frog algorithm integration.

    Args:
        SIMULATION : Simulation object from OpenMM. Minimized and equilibrated.
        INTEGRATOR : Group of integrators.
    """
    
    #  Set the new reporters.
    print("Setting new reporters cMD...")
    SIMULATION.reporters.append(PDBReporter('md_0_1.pdb', 10000))
    
    #  Stdout output.
    SIMULATION.reporters.append(StateDataReporter(stdout, 1000, step=True,
        potentialEnergy=True, temperature=True, volume=True, density=True))
    
    #  Write log file.
    SIMULATION.reporters.append(StateDataReporter("md_0_1_md_log.txt", 100, step=True,
        potentialEnergy=True, temperature=True, volume=True, density=True))
    
    #  Write the trajectory.
    SIMULATION.reporters.append(DCDReporter("md_0_1.dcd",100))
    
    #  Set the right integrator.
    INTEGRATOR.setCurrentIntegrator(1)
    
    #  Run the simulation.
    SIMULATION.step(500000)
    
    return SIMULATION

def accelerated_md(SIMULATION,INTEGRATOR) :
    """Run a aMD using the aMD integrator.

    Args:
        SIMULATION : Simulation object from OpenMM. Minimized and equilibrated.
        INTEGRATOR : Group of integrators.
    """

    #  Set the new reporters.
    print("Setting new reporters aMD...")
    SIMULATION.reporters.append(PDBReporter('Amd_0_1.pdb', 10000))
    
    #  Stdout output.
    SIMULATION.reporters.append(StateDataReporter(stdout, 1000, step=True,
        potentialEnergy=True, temperature=True, volume=True, density=True))
    
    #  Write log file.
    SIMULATION.reporters.append(StateDataReporter("Amd_0_1_md_log.txt", 100, step=True,
        potentialEnergy=True, temperature=True, volume=True, density=True))
    
    #  Write the trajectory.
    SIMULATION.reporters.append(DCDReporter("Amd_0_1.dcd",100))

    #  Set the right integrator.
    INTEGRATOR.setCurrentIntegrator(2)
    
    #  Run the simulation.
    SIMULATION.step(500000)

    return SIMULATION

if __name__ == "__main__" :
    
    SIMULATION, INTEGRATOR, SYSTEM = setup_system(PDB_FILE)
    SIMULATION = minimization_step(SIMULATION)
    SIMULATION = NVT_equilibration(SIMULATION,INTEGRATOR)
    SIMULATION, SYSTEM = NPT_equilibration(SIMULATION, INTEGRATOR, SYSTEM)
    SIMULATION = classic_md(SIMULATION,INTEGRATOR)
    SIMULATION = accelerated_md(SIMULATION, INTEGRATOR)

    print("######### MD IS DONE #########")
