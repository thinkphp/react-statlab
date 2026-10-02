# StatLab

A statistical analysis app with a React + TypeScript interface, interactive Recharts visualizations, and a Python/FastAPI backend powered by NumPy and SciPy.

## Getting started

Requirements: Node.js 20.19+ and Python 3.10+.

```bash
npm install
python3 -m venv .venv
. .venv/bin/activate
pip install -r backend/requirements.txt
```

Start the API in one terminal:

```bash
uvicorn backend.main:app --reload --port 8000
```

Start the frontend in another terminal:

```bash
npm run dev
```

Open the Vite URL shown in the terminal (default: `http://localhost:5173`). Vite proxies `/api` requests to FastAPI at `127.0.0.1:8000`. OpenAPI documentation is available at `http://127.0.0.1:8000/docs`.

## Features

- Paste CSV/TSV data with headers, enter values manually for two variables, or generate random data.
- Descriptive statistics for each column: mean, median, mode, sample variance and standard deviation, quartiles, IQR, and outliers (1.5 × IQR).
- Interactive histogram and box plot, plus Pearson/Spearman correlation matrices.
- Simple and multiple OLS linear regression with coefficients, R², adjusted R², standard error, predicted values, and residuals.
- PDF/PMF and CDF visualizations for normal, binomial, and Poisson distributions.
- Welch's two-sided t-test and chi-square goodness-of-fit test with optional expected counts.

Data is processed by the local API and is not stored by the app. Run backend tests with:

```bash
cd backend && python3 -m unittest -v
```
