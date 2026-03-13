import numpy as np
import numpyro
from jax import numpy as jnp
from pyrenew.deterministic import DeterministicVariable

from pyrenew_multisignal.hew import PyrenewHEWData, PyrenewHEWParam, build_pyrenew_hew_model


def test_build_and_sample_pyrenew_hew_model():
    params = PyrenewHEWParam(
        population_size=1_000_000,
        pop_fraction=jnp.array([0.6, 0.4]),
        generation_interval_pmf=jnp.array([0.25, 0.25, 0.25, 0.25]),
        inf_to_hosp_admit_lognormal_loc=1.1,
        inf_to_hosp_admit_lognormal_scale=0.4,
        inf_to_hosp_admit_pmf=jnp.array([0.05, 0.2, 0.35, 0.3, 0.1]),
        right_truncation_pmf=jnp.array([0.5, 0.3, 0.2]),
    )

    priors = {
        "i0_first_obs_n_rv": DeterministicVariable("i0_first_obs_n", 1e-6),
        "log_r_mu_intercept_rv": DeterministicVariable("log_r_mu_intercept", 0.08),
        "autoreg_rt_rv": DeterministicVariable("autoreg_rt", 0.4),
        "eta_sd_rv": DeterministicVariable("eta_sd", 0.1),
        "inf_feedback_strength_rv": DeterministicVariable("inf_feedback_strength", -2.0),
        "autoreg_rt_subpop_rv": DeterministicVariable("autoreg_rt_subpop", 0.3),
        "sigma_rt_rv": DeterministicVariable("sigma_rt", 0.1),
        "sigma_i_first_obs_rv": DeterministicVariable("sigma_i_first_obs", 0.1),
        "offset_ref_logit_i_first_obs_rv": DeterministicVariable(
            "offset_ref_logit_i_first_obs", 0.0
        ),
        "offset_ref_log_rt_rv": DeterministicVariable("offset_ref_log_rt", 0.0),
        "p_ed_visit_mean_rv": DeterministicVariable("p_ed_visit_mean", 0.0),
        "p_ed_visit_w_sd_rv": DeterministicVariable("p_ed_visit_w_sd", 0.1),
        "autoreg_p_ed_visit_rv": DeterministicVariable("autoreg_p_ed_visit", 0.4),
        "ed_visit_wday_effect_rv": DeterministicVariable(
            "ed_visit_wday_effect", jnp.ones(7)
        ),
        "ed_neg_bin_concentration_rv": DeterministicVariable(
            "ed_neg_bin_concentration", 10.0
        ),
        "hosp_admit_neg_bin_concentration_rv": DeterministicVariable(
            "hosp_admit_neg_bin_concentration", 10.0
        ),
        "ihr_rv": DeterministicVariable("ihr", 0.01),
        "ihr_rel_iedr_rv": DeterministicVariable("ihr_rel_iedr", 0.8),
        "t_peak_rv": DeterministicVariable("t_peak", 4.0),
        "duration_shed_after_peak_rv": DeterministicVariable(
            "duration_shed_after_peak", 10.0
        ),
        "log10_genome_per_inf_ind_rv": DeterministicVariable(
            "log10_genome_per_inf_ind", 8.0
        ),
        "mode_sigma_ww_site_rv": DeterministicVariable("mode_sigma_ww_site", 0.5),
        "sd_log_sigma_ww_site_rv": DeterministicVariable("sd_log_sigma_ww_site", 0.1),
        "mode_sd_ww_site_rv": DeterministicVariable("mode_sd_ww_site", 0.1),
        "max_shed_interval": 14,
        "ww_ml_produced_per_day": 1_000_000.0,
    }

    model = build_pyrenew_hew_model(
        priors=priors,
        params=params,
        fit_ed_visits=False,
        fit_hospital_admissions=False,
        fit_wastewater=False,
    )

    data = PyrenewHEWData(
        n_ed_visits_data_days=14,
        first_ed_visits_date=np.datetime64("2024-01-01"),
    )

    with numpyro.handlers.seed(rng_seed=42):
        samples = model.sample(
            data=data,
            sample_ed_visits=False,
            sample_hospital_admissions=False,
            sample_wastewater=False,
        )

    assert samples["ed_visits"] is None
    assert samples["hospital_admissions"] is None
    assert samples["site_level_wastewater_conc"] is None
    assert samples["population_level_latent_wastewater_conc"] is None
