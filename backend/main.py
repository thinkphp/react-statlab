from collections import Counter
from math import ceil, isfinite, sqrt
from typing import Literal

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from scipy import stats

app = FastAPI(title="StatLab API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


class AnalyzeRequest(BaseModel):
    data: dict[str, list[float]]
    target: str | None = None
    predictors: list[str] = Field(default_factory=list)


class DistributionRequest(BaseModel):
    type: Literal["normal", "binomial", "poisson"]
    mean: float | None = None
    std: float | None = None
    n: int | None = None
    p: float | None = None
    lambda_: float | None = Field(default=None, alias="lambda")

    class Config:
        populate_by_name = True


class TestRequest(BaseModel):
    type: Literal["ttest", "chi_square"]
    values_a: list[float]
    values_b: list[float] | None = None
    expected: list[float] | None = None


def _number(value: float | np.floating | None) -> float | None:
    if value is None or not isfinite(float(value)):
        return None
    return float(value)


def _validate_dataset(data: dict[str, list[float]]) -> tuple[list[str], dict[str, np.ndarray], int]:
    if not data:
        raise HTTPException(status_code=422, detail="Add at least one numeric column.")
    if len(data) > 20:
        raise HTTPException(status_code=422, detail="A maximum of 20 columns is allowed.")
    names = list(data)
    lengths = {len(values) for values in data.values()}
    if len(lengths) != 1:
        raise HTTPException(status_code=422, detail="All columns must have the same number of values.")
    rows = lengths.pop()
    if rows < 2:
        raise HTTPException(status_code=422, detail="At least two observations are required.")
    if rows > 10000:
        raise HTTPException(status_code=422, detail="A maximum of 10,000 rows is allowed.")
    if any(not name.strip() for name in names):
        raise HTTPException(status_code=422, detail="Column names cannot be empty.")
    try:
        arrays = {name: np.asarray(values, dtype=float) for name, values in data.items()}
    except (TypeError, ValueError) as error:
        raise HTTPException(status_code=422, detail="All values must be numeric.") from error
    if any(not np.isfinite(values).all() for values in arrays.values()):
        raise HTTPException(status_code=422, detail="Missing or infinite values must be fixed before analysis.")
    return names, arrays, rows


def _describe(values: np.ndarray) -> dict:
    ordered = np.sort(values)
    q1, median, q3 = np.percentile(ordered, [25, 50, 75])
    iqr = q3 - q1
    lower_fence = q1 - 1.5 * iqr
    upper_fence = q3 + 1.5 * iqr
    outlier_mask = (ordered < lower_fence) | (ordered > upper_fence)
    inliers = ordered[~outlier_mask]
    frequencies = Counter(values.tolist())
    highest_frequency = max(frequencies.values())
    modes = sorted(value for value, count in frequencies.items() if count == highest_frequency) if highest_frequency > 1 else []
    return {
        "count": int(values.size),
        "mean": float(np.mean(values)),
        "median": float(median),
        "mode": [float(value) for value in modes],
        "variance": float(np.var(values, ddof=1)),
        "std_dev": float(np.std(values, ddof=1)),
        "min": float(ordered[0]),
        "max": float(ordered[-1]),
        "q1": float(q1),
        "q3": float(q3),
        "iqr": float(iqr),
        "lower_fence": float(lower_fence),
        "upper_fence": float(upper_fence),
        "lower_whisker": float(inliers[0]) if inliers.size else float(ordered[0]),
        "upper_whisker": float(inliers[-1]) if inliers.size else float(ordered[-1]),
        "outliers": [float(value) for value in ordered[outlier_mask]],
    }


def _histogram(values: np.ndarray) -> dict:
    bin_count = min(30, max(1, int(ceil(np.log2(values.size) + 1))))
    counts, edges = np.histogram(values, bins=bin_count)
    return {
        "bins": [
            {
                "start": float(edges[index]),
                "end": float(edges[index + 1]),
                "count": int(count),
                "label": f"{edges[index]:.2f}–{edges[index + 1]:.2f}",
            }
            for index, count in enumerate(counts)
        ]
    }


def _correlations(names: list[str], arrays: dict[str, np.ndarray], method: str) -> dict:
    output: dict[str, dict[str, float | None]] = {}
    for row in names:
        output[row] = {}
        for column in names:
            if row == column:
                output[row][column] = 1.0
                continue
            result = stats.pearsonr(arrays[row], arrays[column]) if method == "pearson" else stats.spearmanr(arrays[row], arrays[column])
            output[row][column] = _number(result.statistic)
    return output


def _regression(arrays: dict[str, np.ndarray], target: str, predictors: list[str]) -> dict | None:
    if not predictors:
        return None
    y = arrays[target]
    x = np.column_stack([arrays[name] for name in predictors])
    design = np.column_stack([np.ones(y.size), x])
    if y.size <= design.shape[1]:
        return None
    coefficients, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
    predicted = design @ coefficients
    residuals = y - predicted
    ss_residual = float(np.sum(residuals**2))
    ss_total = float(np.sum((y - np.mean(y)) ** 2))
    r_squared = 1.0 if ss_total == 0 and ss_residual == 0 else (0.0 if ss_total == 0 else 1 - ss_residual / ss_total)
    observations = int(y.size)
    predictor_count = len(predictors)
    adjusted_r_squared = 1 - (1 - r_squared) * (observations - 1) / (observations - predictor_count - 1)
    residual_std_error = sqrt(ss_residual / (observations - predictor_count - 1))
    return {
        "target": target,
        "predictors": predictors,
        "intercept": float(coefficients[0]),
        "coefficients": {name: float(coefficients[index + 1]) for index, name in enumerate(predictors)},
        "r_squared": float(r_squared),
        "adjusted_r_squared": float(adjusted_r_squared),
        "residual_std_error": float(residual_std_error),
        "observations": observations,
        "residuals": [
            {
                "predicted": float(predicted[index]),
                "actual": float(y[index]),
                "residual": float(residuals[index]),
                "predictors": {name: float(arrays[name][index]) for name in predictors},
            }
            for index in range(observations)
        ],
    }


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok", "engine": "python"}


@app.post("/api/analyze")
def analyze(request: AnalyzeRequest) -> dict:
    names, arrays, rows = _validate_dataset(request.data)
    target = request.target
    if target is not None and target not in names:
        raise HTTPException(status_code=422, detail="The outcome variable does not exist in the dataset.")
    predictors = request.predictors
    if len(set(predictors)) != len(predictors):
        raise HTTPException(status_code=422, detail="Predictors cannot be duplicated.")
    if any(name not in names for name in predictors):
        raise HTTPException(status_code=422, detail="One or more predictors do not exist in the dataset.")
    if target in predictors:
        raise HTTPException(status_code=422, detail="The outcome variable cannot also be a predictor.")
    return {
        "rows": rows,
        "columns": names,
        "statistics": {name: _describe(arrays[name]) for name in names},
        "histograms": {name: _histogram(arrays[name]) for name in names},
        "correlations": _correlations(names, arrays, "pearson"),
        "spearman_correlations": _correlations(names, arrays, "spearman"),
        "regression": _regression(arrays, target, predictors) if target else None,
    }


@app.post("/api/distribution")
def distribution(request: DistributionRequest) -> dict:
    if request.type == "normal":
        mean = request.mean if request.mean is not None else 0.0
        std = request.std if request.std is not None else 1.0
        if std <= 0:
            raise HTTPException(status_code=422, detail="Standard deviation must be positive.")
        x_values = np.linspace(mean - 4 * std, mean + 4 * std, 121)
        points = [{"x": float(x), "density": float(stats.norm.pdf(x, mean, std)), "cdf": float(stats.norm.cdf(x, mean, std))} for x in x_values]
        label = f"Normal distribution (μ = {mean:g}, σ = {std:g})"
    elif request.type == "binomial":
        n = request.n if request.n is not None else 20
        probability = request.p if request.p is not None else 0.5
        if not 1 <= n <= 200:
            raise HTTPException(status_code=422, detail="n must be between 1 and 200.")
        if not 0 <= probability <= 1:
            raise HTTPException(status_code=422, detail="Probability p must be between 0 and 1.")
        x_values = np.arange(n + 1)
        points = [{"x": int(x), "density": float(stats.binom.pmf(x, n, probability)), "cdf": float(stats.binom.cdf(x, n, probability))} for x in x_values]
        label = f"Binomial distribution (n = {n}, p = {probability:g})"
    else:
        rate = request.lambda_ if request.lambda_ is not None else 5.0
        if not 0 < rate <= 100:
            raise HTTPException(status_code=422, detail="Rate λ must be greater than 0 and at most 100.")
        x_values = np.arange(min(200, max(12, int(ceil(rate + 5 * sqrt(rate))) + 1)))
        points = [{"x": int(x), "density": float(stats.poisson.pmf(x, rate)), "cdf": float(stats.poisson.cdf(x, rate))} for x in x_values]
        label = f"Poisson distribution (λ = {rate:g})"
    return {"type": request.type, "label": label, "points": points}


@app.post("/api/test")
def statistical_test(request: TestRequest) -> dict:
    if request.type == "ttest":
        if request.values_b is None or len(request.values_a) < 2 or len(request.values_b) < 2:
            raise HTTPException(status_code=422, detail="The t-test requires at least two values in each sample.")
        result = stats.ttest_ind(request.values_a, request.values_b, equal_var=False)
        degrees_of_freedom = _number(result.df)
        label = "Welch's t-test (two independent samples)"
        statistic = _number(result.statistic)
        p_value = _number(result.pvalue)
    else:
        observed = np.asarray(request.values_a, dtype=float)
        if observed.size < 2 or np.any(observed < 0) or not np.isfinite(observed).all():
            raise HTTPException(status_code=422, detail="Enter at least two non-negative observed counts.")
        expected = np.asarray(request.expected, dtype=float) if request.expected is not None else np.full(observed.size, observed.sum() / observed.size)
        if expected.size != observed.size or np.any(expected <= 0) or not np.isfinite(expected).all():
            raise HTTPException(status_code=422, detail="Expected counts must be positive and match the observed counts.")
        if not np.isclose(observed.sum(), expected.sum()):
            raise HTTPException(status_code=422, detail="The sum of observed counts must equal the sum of expected counts.")
        result = stats.chisquare(observed, expected)
        degrees_of_freedom = float(observed.size - 1)
        label = "Chi-square goodness-of-fit test"
        statistic = _number(result.statistic)
        p_value = _number(result.pvalue)

    if statistic is None or p_value is None or degrees_of_freedom is None:
        raise HTTPException(status_code=422, detail="The test is undefined for these values; check that the samples have variation.")
    significant = p_value < 0.05
    interpretation = (
        "There is statistical evidence against the null hypothesis (H₀)."
        if significant
        else "There is insufficient evidence to reject the null hypothesis (H₀)."
    )
    return {
        "type": request.type,
        "label": label,
        "statistic": statistic,
        "p_value": p_value,
        "degrees_of_freedom": degrees_of_freedom,
        "significant": significant,
        "interpretation": interpretation,
    }
