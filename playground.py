from utils.Dataclasses import Study
from dataclasses import asdict

if __name__ == "__main__":
    study = Study.from_json('LiteratureReview/isotherm_kinetics.json', "jiangAdsorptionLithiumIons2020")
    
    curves = study.column_experiments.breakthrough_curves.curves
    
    print(study.column_experiments.superficial_velocity_si())
    print(study.column_experiments.interstitial_velocity_si())
    print(study.particle_reynolds())
    print(study.schmidt_number(T=303))
    print(study.particle_peclet_number(curves[2]))