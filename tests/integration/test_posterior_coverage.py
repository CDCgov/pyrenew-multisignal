"""
Integration test: MCMC trajectory recovery on pyrenew synthetic H+E data.

Uses pyrenew.datasets synthetic data (120-day CA COVID-19 trajectory) with
known ground-truth parameters. Fits  Pyrenew-HE model and checks
90% posterior CI covers the true R(t) for >= 80% of time points

"""

import runpy
from pathlib import Path

import arviz as az
import jax.random as jr
import numpy as np
import numpyro
import polars as pl
import pytest
from pyrenew.datasets import (
    load_synthetic_daily_ed_visits,
    load_synthetic_daily_infections,
    load_synthetic_true_parameters,
    load_synthetic_weekly_hospital_admissions,
)

from pyrenew_multisignal.hew import (
    PyrenewHEWData,
    PyrenewHEWParam,
    build_pyrenew_hew_model,
)

numpyro.set_host_device_count(2)

TESTS_DIR = Path("tests/integration")
MODEL_PARAMS_PATH = TESTS_DIR / "model_params.json"
PRIORS_PATH = TESTS_DIR / "priors.py"

NUM_WARMUP = 500
NUM_SAMPLES = 500
NUM_CHAINS = 2
COVERAGE_THRESHOLD = 0.80
MCMC_SEED = 123


def _build_nssp_data() -> pl.DataFrame:
    """Reshape synthetic daily ED visit data to the schema expected by PyrenewHEWData."""
    return (
        load_synthetic_daily_ed_visits()
        .rename({"ed_visits": "observed_ed_visits"})
        .with_columns(
            pl.col("date").str.strptime(pl.Date, "%Y-%m-%d"),
            pl.col("observed_ed_visits").cast(pl.Float64),
            pl.lit(0.0).alias("other_ed_visits"),
            pl.lit("train").alias("data_type"),
        )
        .select(
            ["date", "geo_value", "observed_ed_visits", "other_ed_visits", "data_type"]
        )
    )


def _build_nhsn_data() -> pl.DataFrame:
    """Reshape synthetic weekly hospital admissions data to the schema expected by PyrenewHEWData."""
    return (
        load_synthetic_weekly_hospital_admissions()
        .rename(
            {
                "week_end": "weekendingdate",
                "location": "jurisdiction",
                "weekly_hosp_admits": "hospital_admissions",
            }
        )
        .with_columns(
            pl.col("weekendingdate").str.strptime(pl.Date, "%Y-%m-%d"),
            pl.col("hospital_admissions").cast(pl.Float64),
            pl.lit("train").alias("data_type"),
        )
        .select(["weekendingdate", "jurisdiction", "hospital_admissions", "data_type"])
    )


@pytest.fixture(scope="module")
def true_params():
    return load_synthetic_true_parameters()


@pytest.fixture(scope="module")
def true_daily_infections():
    return load_synthetic_daily_infections()


@pytest.fixture(scope="module")
def training_data(true_params) -> PyrenewHEWData:
    """PyrenewHEWData built from the pyrenew synthetic datasets."""
    return PyrenewHEWData(
        nssp_training_data=_build_nssp_data(),
        nhsn_training_data=_build_nhsn_data(),
        population_size=true_params["population"],
        right_truncation_offset=None,
    )


@pytest.fixture(scope="module")
def fitted_model(training_data):
    """Fit Pyrenew-HE model to the synthetic data and return the fitted model."""
    params = PyrenewHEWParam.from_json(MODEL_PARAMS_PATH)
    priors = runpy.run_path(str(PRIORS_PATH))

    model = build_pyrenew_hew_model(
        priors=priors,
        params=params,
        fit_hospital_admissions=True,
        fit_ed_visits=True,
    )

    model.run(
        num_warmup=NUM_WARMUP,
        num_samples=NUM_SAMPLES,
        rng_key=jr.key(MCMC_SEED),
        mcmc_args={"num_chains": NUM_CHAINS, "progress_bar": False},
        data=training_data,
        sample_hospital_admissions=True,
        sample_ed_visits=True,
    )
    return model


@pytest.fixture(scope="module")
def idata(fitted_model):
    """Convert fitted MCMC to ArviZ InferenceData."""
    return az.from_numpyro(fitted_model.mcmc)


@pytest.fixture(scope="module")
def true_rt(true_daily_infections):
    """Extract true R(t) trajectory from synthetic infections data."""
    return true_daily_infections["true_rt"].to_numpy()


@pytest.mark.integration
def test_rt_posterior_covers_truth(idata, true_rt):
    """Check that 90% credible interval for R(t) covers the true value
    for at least 80% of time points."""
    rt_posterior = idata.posterior["rt"]
    rt_q05 = rt_posterior.quantile(0.05, dim=["chain", "draw"]).values
    rt_q95 = rt_posterior.quantile(0.95, dim=["chain", "draw"]).values

    covered = (true_rt >= rt_q05) & (true_rt <= rt_q95)
    coverage = float(np.mean(covered))

    assert coverage >= COVERAGE_THRESHOLD, (
        f"R(t) 90% CI coverage was {coverage:.1%}, expected >= {COVERAGE_THRESHOLD:.1%}"
    )
