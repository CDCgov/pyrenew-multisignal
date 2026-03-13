# numpydoc ignore=GL08

import jax.numpy as jnp
import numpy as np
import pytest
from jax.typing import ArrayLike
from pyrenew.latent import compute_infections_from_rt

from pyrenew_multisignal.hew import (
    InfectionsWithSusceptibleDepletion,
)

compute_infections_with_susceptible_depletion = (
    InfectionsWithSusceptibleDepletion.compute_infections_with_susceptible_depletion
)


def _infection_w_sus_depletion(
    gen_int: ArrayLike,
    Rt: ArrayLike,
    I0: ArrayLike,
    pop: ArrayLike,
    S0: ArrayLike,
) -> tuple:
    """
    Calculate the infections with susceptible depletion.
    Parameters
    ----------
    gen_int
        Generation interval.
    Rt
        Reproduction number.
    I0
        Initial infections.
    pop
        Population size.
    S0
        Initial susceptible population.

    Returns
    -------
    tuple
    """
    T = len(Rt)
    len_gen = len(gen_int)
    Inf = np.pad(I0, (0, T))
    S = np.pad(np.atleast_1d(S0), (0, T))
    R_adj = np.array(Rt)

    for t in range(T):
        infectiousness = jnp.dot(Inf[t : t + len_gen], jnp.flip(gen_int))
        Inf[t + len_gen] = S[t] * (-jnp.expm1(-Rt[t] * infectiousness / pop))
        S[t + 1] = S[t] - Inf[t + len_gen]
        R_adj[t] = np.where(infectiousness > 0, Inf[t + len_gen] / infectiousness, 0)

    return {
        "post_initialization_infections": Inf[-T:],
        "rt": R_adj,
    }


@pytest.mark.parametrize(
    "I0, gen_int, Rt, pop, S0",
    [
        (2 * jnp.ones(4), jnp.ones(4) / 4, jnp.ones(10), 100000, 10000.0),
    ],
)
def test_infections_with_sus_depletion(I0, gen_int, Rt, pop, S0):
    """
    Test the InfectionsWithSusceptibleDepletion class
    """
    Inf_w_sus_depletion = InfectionsWithSusceptibleDepletion(
        name="test_inf_sus_depletion"
    )

    res = Inf_w_sus_depletion.sample(
        Rt=Rt, I0=I0, gen_int=gen_int, S0=S0, population=pop
    )

    res_bf = _infection_w_sus_depletion(gen_int=gen_int, Rt=Rt, I0=I0, pop=pop, S0=S0)
    assert jnp.allclose(
        res.post_initialization_infections, res_bf["post_initialization_infections"]
    )
    assert jnp.allclose(res.rt, res_bf["rt"])


@pytest.mark.parametrize(
    "S0,population,error_match",
    [
        (
            jnp.array([10], dtype=float),
            jnp.array([10, 10]),
            "S0 must match Rt batch shape exactly",
        ),
        (
            jnp.array([10, 10], dtype=float),
            jnp.array([10]),
            "population must match Rt batch shape exactly",
        ),
        (
            jnp.array([10, 10], dtype=float),
            jnp.array([5, 10]),
            "Susceptible cannot be greater than population",
        ),
    ],
)
def test_infections_with_sus_depletion_invalid_inputs(S0, population, error_match):
    """
    Test the InfectionsWithSusceptibleDepletion class cannot
    be sampled when Rt, S0, and population have invalid input shapes
    """
    I0 = jnp.array([[5.0, 0.2]])
    gen_int = jnp.ones(1)
    Rt = jnp.ones((5, 2))

    Inf_w_sus_depletion = InfectionsWithSusceptibleDepletion(
        name="test_inf_sus_depletion"
    )

    with pytest.raises(ValueError, match=error_match):
        Inf_w_sus_depletion.sample(
            Rt=Rt,
            I0=I0,
            gen_int=gen_int,
            S0=S0,
            population=population,
        )


@pytest.mark.parametrize(
    ["I0", "gen_int", "Rt_raw", "S0"],
    [
        [
            jnp.array([[5.0, 0.2]]),
            jnp.array([1.0]),
            jnp.ones((5, 2)),
            jnp.array([10**7, 10**6], dtype=float),
        ],
        [
            2 * jnp.ones(4),
            jnp.ones(4) / 4,
            jnp.ones(10),
            10**6,
        ],
    ],
)
def test_compute_infections_with_susceptible_depletion(I0, gen_int, Rt_raw, S0):
    """
    Test implementation of susceptible depletion
    when initial susceptible population is large
    enough that depletion does not affect infections.
    """
    (
        infs_sus_depletion,
        Rt_adj,
    ) = compute_infections_with_susceptible_depletion(I0, Rt_raw, gen_int, S0, S0)

    assert jnp.allclose(
        compute_infections_from_rt(I0, Rt_raw, gen_int),
        infs_sus_depletion,
        rtol=1e-4,
    )

    assert jnp.allclose(Rt_adj, Rt_raw, rtol=1e-4)

    assert jnp.all(jnp.sum(infs_sus_depletion, axis=0) <= S0)

    return None


@pytest.mark.parametrize(
    ["S0"],
    [
        [
            10**6,
        ],
        [
            10,
        ],
    ],
)
def test_compute_infections_with_susceptible_depletion_zero_rt(S0):
    """
    Test implementation of susceptible depletion
    when Rt is zero, so that infections and adjusted
    Rt should be zero and susceptible population
    should not deplete.
    """

    I0 = 2 * jnp.ones(4)
    gen_int = jnp.ones(4) / 4
    Rt_raw = jnp.zeros(10)

    infections, Rt_adjusted = compute_infections_with_susceptible_depletion(
        I0, Rt_raw, gen_int, S0, S0
    )

    assert jnp.allclose(infections, jnp.zeros_like(Rt_raw))
    assert jnp.allclose(Rt_adjusted, jnp.zeros_like(Rt_raw))

    return None


@pytest.mark.parametrize(
    ["S0", "pop"],
    [
        [
            10,
            10**6,
        ],
    ],
)
def test_compute_infections_with_susceptible_depletion_small_S0(S0, pop):
    """
    Test implementation of susceptible depletion
    when initial susceptible population is small
    enough that susceptible depletion aborts infections.
    """
    I0 = 2 * jnp.ones(4)
    gen_int = jnp.ones(4) / 4
    Rt_raw = jnp.ones(10)

    infections, _ = compute_infections_with_susceptible_depletion(
        I0, Rt_raw, gen_int, S0, pop
    )

    assert jnp.allclose(infections[-1], 0)

    return None
