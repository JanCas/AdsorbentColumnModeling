import numpy as np
import do_mpc
from utils.Dataclasses import Study, BreakthroughCurve
from casadi import vertcat
from tqdm import tqdm
import matplotlib.pyplot as plt
from utils.ResultsDataclasses import SimulationResult
import pandas as pd

num_nodes=25

def model(num_nodes: int, study: Study, curve: BreakthroughCurve) -> do_mpc.model.Model:
    '''
        Create a model for a fixed bed adsorption column.
        The model includes advection but not dispersion of the fluid phase.
    '''

    dx = study.column_experiments.column_properties.Length_si / num_nodes
    (D_ax,), _, _ = study.axial_dispersion_coefficient(curve)

    model = do_mpc.model.Model('continuous')

    C_Li = model.set_variable('_x', 'C_Li', shape=(num_nodes,1)) # liquid phase concentration in mol/m^3
    n_i = model.set_variable('_x', 'n_i', shape=(num_nodes,1)) # adsorbed phase concentration in mol/kg

    kinetics_exp = study.get_kinetics_experiment_from_curve(curve)

    dn_i_dt = kinetics_exp.kinetics_params.k2_si * (study.isotherm.isotherm_fit.q_eq_si(C_Li, kinetics_exp.T) - n_i)**2
    model.set_rhs('n_i', dn_i_dt)

    C_up = vertcat(study.column_experiments.influent_concentration_si(curve), C_Li[:-1])  # Upstream concentration with boundary condition
    dC_Li_dt = - study.column_experiments.interstitial_velocity_si(curve) / dx * (C_Li - C_up) - (1 - study.column_experiments.column_properties.porosity) / study.column_experiments.column_properties.porosity * study.sorbent_properties.density * dn_i_dt

    model.set_rhs('C_Li', dC_Li_dt)
    model.setup()
    return model

if __name__ == "__main__":
    s = Study.from_json('LiteratureReview/isotherm_kinetics.json', "jiangAdsorptionLithiumIons2020")

    simulation_results = []


    fig, ax = plt.subplots(1,len(s.column_experiments.breakthrough_curves.curves), figsize=(8,4), sharey=True)
    fig.suptitle('Breakthrough Curves Simulation vs Experimental Data')


    for curve, axes in tqdm(zip(s.column_experiments.breakthrough_curves.curves, ax), desc="Curves", position=0, total=len(s.column_experiments.breakthrough_curves.curves)):
        
        sr = SimulationResult(
            curve_uuid=curve.uuid,
            Reynolds_number=s.particle_reynolds(curve)[0],
            Peclet_number=s.particle_peclet_number(curve)[0],
            Schmidt_number=s.schmidt_number(curve)
        )
        simulation_results.append(sr)


        m = model(num_nodes, s, curve)

        simulator = do_mpc.simulator.Simulator(m)
        simulator.set_param(t_step=1.0)  # [s] time step for

        simulator.setup()
        simulator.reset_history()

        simulator.x0['C_Li'] = np.zeros((num_nodes,1)) + .001
        simulator.x0['n_i'] = np.zeros((num_nodes,1)) + .001
        for i in tqdm(range(int(curve.run_length_hours * 3600)), desc=f"Simulating {curve.flowrate} BV/h", position=1, leave=False):
            simulator.make_step()  # Advance the simulation by one time step




        end_node_concentration = simulator.data['_x', 'C_Li'][:,-1]
        ratio = end_node_concentration / s.column_experiments.influent_concentration_si(curve)
        axes.plot(simulator.data['_time'] / 3600 * curve.flowrate, ratio, label='Model')
        axes.plot(curve.BV, curve.C_out_over_C_in, 'o', label='Experimental')
        axes.set_xlabel('BV')
        axes.set_ylabel(r'$\frac{C_{out}}{C_{in}}$')
        axes.set_title(f'{curve.flowrate} BV/h, {s.column_experiments.superficial_velocity_si(curve)[0]*1000:.2e} L/s')
        axes.legend()
        axes.grid()

    pd.DataFrame(simulation_results).to_csv('Results/simulation_results.csv', index=False) 
    
    fig.tight_layout()
    fig.show()
    fig.savefig('Results/breakthrough_curves_simulation_vs_experimental.png', dpi=1000)
    fig.savefig('Results/breakthrough_curves_simulation_vs_experimental.svg')
    input("End of simulation, press Enter to exit...")