import jax.numpy as jnp
import numpyro
from pyrenew.deterministic import DeterministicVariable

from pyrenew_multisignal.hew import EDVisitObservationProcess


def test_EDVisitObservationProcess_instantiation_and_sampling():
    """
    Test that EDVisitObservationProcess can be instantiated and sampled.
    This catches issues like ARProcess being called without a required
    name argument in __init__ or sample().
    """
    p_ed_mean_rv = DeterministicVariable("p_ed_mean", 0.0)
    p_ed_w_sd_rv = DeterministicVariable("p_ed_w_sd", 0.1)
    autoreg_p_ed_rv = DeterministicVariable("autoreg_p_ed", 0.4)
    ed_wday_effect_rv = DeterministicVariable(
        "ed_wday_effect", jnp.ones(7) / 7
    )
    inf_to_ed_rv = DeterministicVariable(
        "inf_to_ed", jnp.array([0.25, 0.25, 0.25, 0.25])
    )
    ed_neg_bin_concentration_rv = DeterministicVariable(
        "ed_neg_bin_concentration", 10.0
    )
    ed_right_truncation_pmf_rv = DeterministicVariable(
        "right_truncation_pmf", jnp.array([0.5, 0.3, 0.2])
    )

    ed_visit_obs_rv = EDVisitObservationProcess(
        p_ed_mean_rv=p_ed_mean_rv,
        p_ed_w_sd_rv=p_ed_w_sd_rv,
        autoreg_p_ed_rv=autoreg_p_ed_rv,
        ed_wday_effect_rv=ed_wday_effect_rv,
        inf_to_ed_rv=inf_to_ed_rv,
        ed_neg_bin_concentration_rv=ed_neg_bin_concentration_rv,
        ed_right_truncation_pmf_rv=ed_right_truncation_pmf_rv,
    )

    with numpyro.handlers.seed(rng_seed=42):
        observed_ed_visits, iedr = ed_visit_obs_rv(
            latent_infections=jnp.ones(30) * 100.0,
            population_size=1_000_000,
            data_observed=None,
            model_t_observed=None,
            model_t_first_latent_infection=0,
        )

    assert observed_ed_visits is not None
    assert iedr is not None
