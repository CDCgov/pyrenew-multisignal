import jax.numpy as jnp
import numpyro
import pytest
from pyrenew.deterministic import DeterministicVariable

from pyrenew_multisignal.hew import (
    PyrenewHEWModel,
)


def test_pyrenew_hew_model_initializes_infection_from_ed_observation(
    latent_infection_process,
):
    model = PyrenewHEWModel(
        population_size=1,
        latent_infection_process_rv=latent_infection_process,
        ed_visit_obs_process_rv=None,
        hosp_admit_obs_process_rv=None,
        wastewater_obs_process_rv=None,
        iedr_rv=DeterministicVariable("iedr", 0.1),
        e_first_obs_n_rv=DeterministicVariable("e_first_obs_n", 0.02),
    )

    with numpyro.handlers.seed(rng_seed=223):
        i0_first_obs_n, iedr, ihr = model.get_initial_infections()

    assert jnp.allclose(i0_first_obs_n, 0.2)
    assert jnp.allclose(iedr, 0.1)


def test_pyrenew_hew_model_errors_without_observation_based_initialization(
    latent_infection_process,
):
    model = PyrenewHEWModel(
        population_size=1,
        latent_infection_process_rv=latent_infection_process,
        ed_visit_obs_process_rv=None,
        hosp_admit_obs_process_rv=None,
        wastewater_obs_process_rv=None,
    )

    with numpyro.handlers.seed(rng_seed=223):
        with pytest.raises(
            ValueError, match="Must provide at least one of `iedr_rv` or `ihr_rv`."
        ):
            model.get_initial_infections()
