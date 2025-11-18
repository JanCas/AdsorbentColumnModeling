from utils.Dataclasses import Study
from dataclasses import asdict
from utils.ResultsDataclasses import SimulationResult
import pandas as pd
from pyEQL import Solution


if __name__ == "__main__":
    study = Study.from_json('LiteratureReview/isotherm_kinetics.json', "jiangAdsorptionLithiumIons2020")
    
    curves = study.column_experiments.breakthrough_curves.curves

    sr = SimulationResult(
        curve_uuid=curves[0].uuid,
        Reynolds_number=study.particle_reynolds(curves[0])[0],
        Peclet_number=study.particle_peclet_number(curves[0])[0],
        Schmidt_number=study.schmidt_number(curves[0])
    )

    print(Solution(temperature='303K').get_property('Li+', 'transport.diffusion_coefficient'))
    
    df = pd.DataFrame([sr])
    print(df)

    print(sr)

    print(study.column_experiments.superficial_velocity_si(curves[0]))
    print(study.column_experiments.interstitial_velocity_si())
    print(study.particle_reynolds())
    print(study.schmidt_number(T=303))
    print(study.particle_peclet_number(T=303))
    print(study.axial_dispersion_coefficient())