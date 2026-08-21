import numpy as np
import pytest

from Model.Multi_Ion_ALLDH.jiang_validation import (
    LI_MW,
    K_PSO_KG_MOL_S,
    STUDY_KEY,
    calibrate_isotherm,
    concentration_from_elemental_li,
    load_jiang,
)
from utils.StreamData import Composition


def test_jiang_projection_and_full_brine_activity():
    study = load_jiang()
    model, fit = calibrate_isotherm(study)
    feed, water_mass_concentration = concentration_from_elemental_li(350.0)
    feed_molality = feed.molality(water_mass_concentration)
    assert STUDY_KEY == "jiangApplicationConcentrationDependentHSDM2020"
    assert study["Isotherm"]["FitType"] == "Sips"
    assert study["Isotherm"]["FitParameters"]["q_max"] == pytest.approx(5.9522)
    assert fit["n_h2o"] == 1.0
    assert fit["projection_r2"] > 0.998
    assert np.all(
        fit["activity_model_mg_zero_mg_g"] < fit["activity_model_mg_g"]
    )
    assert feed.c_LiCl == pytest.approx(350.0 / LI_MW)
    assert water_mass_concentration == pytest.approx(859.127762995719)
    assert float(model(feed_molality) * LI_MW) == pytest.approx(5.4956203966)
    assert study["KineticsExperiments"] == []
    columns = study["ColumnExperiments"]
    assert [curve["height"] for curve in columns["HeightCurves"]["Curves"]] == [
        0.3,
        0.6,
        1.0,
    ]
    assert [
        curve["influent_concentration"]
        for curve in columns["InfluentConcentrationCurves"]["Curves"]
    ] == [300, 350, 400]
    assert K_PSO_KG_MOL_S == pytest.approx(1.00148162948e-4)
    assert "HSDMParameters" not in study["ColumnExperiments"]
    assert float(
        model(Composition(feed_molality.m_LiCl, 0.0, 0.0))
    ) != pytest.approx(
        float(model(feed_molality))
    )
