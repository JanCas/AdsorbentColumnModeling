from do_mpc.model import Model
from ..utils.Dataclasses import Study
from casadi import vertcat

def create_column_model(num_nodes: int, study: Study) -> Model:
    """
    Create a model for a fixed bed adsorption column.
    The model includes advection but not dispersion of the fluid phase.
    """

    dx = study.column_parameters.Length / num_nodes

    model = Model('continuous')

    C_Li = model.set_variable('_x', 'C_Li', shape=(num_nodes,1)) # liquid phase concentration in mol/m^3
    n_i = model.set_variable('_x', 'n_i', shape=(num_nodes,1)) # adsorbed phase concentration in mol/kg

    dn_i_dt = study.kinetics_experiments[0].kinetics_params.k2_si * (study.isotherm.isotherm_fit.q_eq_si(C_Li, study.kinetics_experiments[0].T) - n_i)**2
    model.set_rhs('n_i', dn_i_dt)

    C_up = vertcat(study.column_parameters.influent_concentration_si, C_Li[:-1])  # Upstream concentration with boundary condition
    dC_Li_dt = - study.column_parameters.interstitial_velocity_si[flowrate_idx] / dx * (C_Li - C_up) - (1 - study.column_parameters.porosity) / study.column_parameters.porosity * study.sorbent_properties.density * dn_i_dt

    model.set_rhs('C_Li', dC_Li_dt)
    model.setup()
    return model

