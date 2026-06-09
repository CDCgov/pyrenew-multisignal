import jax.numpy as jnp
import pytest
from pyrenew.deterministic import DeterministicVariable

from pyrenew_multisignal.hew import LatentInfectionProcess


@pytest.fixture
def latent_infection_process():
    return LatentInfectionProcess(
        s0_rv=DeterministicVariable("s0", 1.0),
        log_r_mu_intercept_rv=DeterministicVariable("log_r_mu_intercept", 0.08),
        autoreg_rt_rv=DeterministicVariable("autoreg_rt", 0.4),
        eta_sd_rv=DeterministicVariable("eta_sd", 0),
        generation_interval_pmf_rv=DeterministicVariable(
            "generation_interval_pmf", jnp.array([0.25, 0.25, 0.25, 0.25])
        ),
        n_initialization_points=10,
    )
