import numpy as np
import do_mpc
from utils.Dataclasses import Study
from casadi import vertcat
from dataclasses import asdict
from pprint import pprint
from tqdm import tqdm
import matplotlib.pyplot as plt

num_nodes=10

def model(num_nodes: int, study: Study) -> do_mpc.model.Model:
    '''
        Create a model for a fixed bed adsorption column.
        The model includes advection but not dispersion of the fluid phase.
    '''

    dx = study.column_parameters.Length / num_nodes

    model = do_mpc.model.Model('continuous')

    C_Li = model.set_variable('_x', 'C_Li', shape=(num_nodes,1)) # liquid phase concentration in mol/m^3
    n_i = model.set_variable('_x', 'n_i', shape=(num_nodes,1)) # adsorbed phase concentration in mol/kg

    dn_i_dt = study.kinetics_experiments[0].kinetics_params.k2_si * (study.isotherm.isotherm_fit.q_eq_si(C_Li, study.kinetics_experiments[0].T) - n_i)**2
    model.set_rhs('n_i', dn_i_dt)

    C_up = vertcat(study.column_parameters.influent_concentration_si, C_Li[:-1])  # Upstream concentration with boundary condition
    dC_Li_dt = - study.column_parameters.interstitial_velocity_si[flowrate_idx] / dx * (C_Li - C_up) - (1 - study.column_parameters.porosity) / study.column_parameters.porosity * study.sorbent_properties.density * dn_i_dt

    model.set_rhs('C_Li', dC_Li_dt)
    model.setup()
    return model

if __name__ == "__main__":
    s = Study.from_json('LiteratureReview/isotherm_kinetics.json', "jiangAdsorptionLithiumIons2020")
    s.isotherm.isotherm_fit.q_eq_si(50, 303)

    m = model(num_nodes, s)

    simulator = do_mpc.simulator.Simulator(m)
    simulator.set_param(t_step=1.0)  # [s] time step for

    simulator.setup()
    simulator.reset_history()

    simulator.x0['C_Li'] = np.zeros((num_nodes,1)) + .001
    simulator.x0['n_i'] = np.zeros((num_nodes,1)) + .001
    for i in tqdm(range(int(1.5*3600))):
        simulator.make_step()  # Advance the simulation by one time step
    

    graphics = do_mpc.graphics.Graphics(simulator.data)

    fig, ax = plt.subplots(2,1, figsize=(8,6), sharex=True)
    graphics.add_line(var_type='_x', var_name='C_Li', axis=ax[0], label='C_Li (mol/m^3)')
    graphics.add_line(var_type='_x', var_name='n_i', axis=ax[1], label='n_i (mol/kg)')
    plt.show(block=False)

    end_node_concentration = simulator.data['_x', 'C_Li'][:,-1]
    ratio = end_node_concentration / s.column_parameters.influent_concentration_si
    plt.figure()
    plt.plot(simulator.data['_time'] / 3600 * s.column_parameters.flowrate[flowrate_idx], ratio, label='Model')
    plt.plot(s.breakthrough_curves[flowrate_idx].BV, s.breakthrough_curves[flowrate_idx].C_out_over_C_in, 'o', label='Experimental')
    plt.xlabel('BV')
    plt.ylabel(r'$\frac{C_{out}}{C_{in}}$')
    plt.title(f'Breakthrough Curve ({s.column_parameters.flowrate[flowrate_idx]} BV/h, {s.column_parameters.superficial_flowrate_si[flowrate_idx]*1000:.2e} L/s)')
    plt.legend()
    plt.grid()
    plt.show(block=False)

    input("End of simulation, press Enter to exit...")