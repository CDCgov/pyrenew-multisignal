import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from pyrenew.arrayutils import repeat_until_n
from pyrenew.deterministic import DeterministicVariable
from pyrenew.latent import (
    InfectionInitializationProcess,
    InitializeInfectionsExponentialGrowth,
)
from pyrenew.math import r_approx_from_R
from pyrenew.process import ARProcess, DifferencedProcess
from pyrenew.randomvariable import DistributionalVariable

from pyrenew_multisignal.hew import (
    InfectionsWithSusceptibleDepletion,
    LatentInfectionProcess,
)


def test_LatentInfectionProcess():
    """
    Tests when there is a single sub-population,
    the hierarchical construct and manual construction
    without the hierarchical component are equivalent.
    """
    e_first_obs_n_rv = DeterministicVariable("e_first_obs_n_rv", 1e-6)
    iedr_rv = DeterministicVariable("iedr_rv", 1e-5)
    s0_rv = DeterministicVariable("s0", 1.0)
    log_r_mu_intercept_rv = DeterministicVariable("log_r_mu_intercept", 0.08)
    eta_sd_rv = DeterministicVariable("eta_sd", 0)
    autoreg_rt_rv = DeterministicVariable("autoreg_rt", 0.4)
    generation_interval_pmf_rv = DeterministicVariable(
        "generation_interval_pmf", jnp.array([0.25, 0.25, 0.25, 0.25])
    )
    initialization_rate_rv = DeterministicVariable(
        "initialization_rate",
        r_approx_from_R(
            jnp.exp(log_r_mu_intercept_rv()),
            generation_interval_pmf_rv(),
            n_newton_steps=4,
        ),
    )
    n_initialization_points = 10
    n_days_post_init = 14

    my_latent_infection_model = LatentInfectionProcess(
        e_first_obs_n_rv=e_first_obs_n_rv,
        iedr_rv=iedr_rv,
        s0_rv=s0_rv,
        log_r_mu_intercept_rv=log_r_mu_intercept_rv,
        autoreg_rt_rv=autoreg_rt_rv,
        eta_sd_rv=eta_sd_rv,  # sd of random walk for ar process,
        generation_interval_pmf_rv=generation_interval_pmf_rv,
        n_initialization_points=n_initialization_points,
    )

    with numpyro.handlers.seed(rng_seed=223):
        latent_inf_w_hierarchical_effects, _, iedr, _ = my_latent_infection_model(
            n_days_post_init=n_days_post_init
        )

        e_first_obs_n = e_first_obs_n_rv()
        i0_first_obs_n = e_first_obs_n / iedr

        # Calculate latent infections without hierarchical dynamics
        i0 = InfectionInitializationProcess(
            "I0_initialization",
            DeterministicVariable("i0_first_obs_n", i0_first_obs_n),
            InitializeInfectionsExponentialGrowth(
                n_initialization_points, initialization_rate_rv, t_pre_init=0
            ),
        )()

        inf_with_susceptible_dep_proc = InfectionsWithSusceptibleDepletion(
            "infections_with_susceptible_dep"
        )

        ar_diff = DifferencedProcess(
            "ar_diff",
            fundamental_process=ARProcess("ar"),
            differencing_order=1,
        )

        rt_init_rate_of_change = DistributionalVariable(
            "rt_init_rate_of_change",
            dist.Normal(0, eta_sd_rv() / jnp.sqrt(1 - jnp.pow(autoreg_rt_rv(), 2))),
        )()

        log_rtu_weekly = ar_diff(
            n=2,
            init_vals=jnp.array(log_r_mu_intercept_rv()),
            autoreg=jnp.array(autoreg_rt_rv()),
            noise_sd=jnp.array(eta_sd_rv()),
            fundamental_process_init_vals=jnp.array(rt_init_rate_of_change),
            noise_name="rtu_weekly_diff_first_diff_ar_process_noise",
        )
        rtu = repeat_until_n(
            data=jnp.exp(log_rtu_weekly),
            n_timepoints=n_days_post_init,
            offset=0,
            period_size=7,
        )
        inf_with_susceptible_dep_proc_sample = inf_with_susceptible_dep_proc(
            Rt=rtu,
            I0=i0,
            gen_int=generation_interval_pmf_rv(),
            S0=s0_rv(),
            population=jnp.array(1.0),
        )

    latent_inf_wo_hierarchical_effects = jnp.concat(
        [
            i0,
            inf_with_susceptible_dep_proc_sample.post_initialization_infections,
        ]
    )

    assert jnp.allclose(
        latent_inf_w_hierarchical_effects, latent_inf_wo_hierarchical_effects
    )
