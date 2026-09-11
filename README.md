# Sentinel Risk Intelligence Platform

![Python](https://img.shields.io/badge/Python-3.13-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.103.2-009688.svg)
![HTMX](https://img.shields.io/badge/HTMX-1.9.10-3D72D7.svg)
![Tailwind](https://img.shields.io/badge/Tailwind_CSS-3.4-38B2AC.svg)
![XGBoost](https://img.shields.io/badge/XGBoost-Enabled-orange.svg)

**Sentinel** is a high-performance, quantitative risk management and surveillance terminal. It provides a unified platform for real-time fraud detection, credit risk modeling, market volatility forecasting, and systemic stress testing, natively integrated with an HTMX + Tailwind dashboard.

## Key Risk Engines

*   **Transaction Surveillance (Fraud):** Real-time XGBoost classifier identifying fraudulent behavior utilizing temporal velocity, amount Z-scores, and categorical encoding.
*   **Loan Default Engine (Credit Risk):** Probability of Default (PD) and Expected Loss (EL) calculations leveraging decision trees on borrower credit data (FICO, DTI).
*   **Portfolio Value at Risk (Market Risk):** 99% VaR and Expected Shortfall calculations utilizing Monte Carlo simulations with Kupiec POF backtesting validation.
*   **Volatility Surface:** Long-run annualized volatility forecasting using GARCH(1,1) Maximum Likelihood Estimation (MLE) and Exponentially Weighted Moving Averages (EWMA).
*   **Scenario Analysis (Stress Testing):** Shock absorption simulations for historical crises (e.g., 2008 GFC, 2022 Rate Shocks) and hypothetical sector crashes.
*   **Obligor 360:** A unified, cross-engine entity search that aggregates an individual borrower's credit, fraud, and exposure risks into a single pane of glass.
*   **Data Forge:** Synthetic data generation pipeline capable of rapidly seeding millions of rows of correlated, stressed financial data for backtesting.

Want to know exactly how the math and architecture work? [Read here](HOW_IT_WORKS.md).

## Limitations

> **Synthetic Data Disclaimer:** Sentinel uses synthetic financial data for demonstration and testing purposes. Model performance should not be interpreted as production performance on real-world banking data. Synthetic data may contain assumptions and patterns that make prediction easier than in real financial environments.

## Architecture & Tech Stack

Sentinel is built to prioritize accuracy, speed, and clean UX without relying on heavy frontend frameworks:

*   **Backend:** Python 3, FastAPI, SQLAlchemy
*   **Quantitative Modeling:** Pandas, Scikit-Learn, XGBoost, SciPy, Arch
*   **Frontend:** HTMX (Server-Side UI rendering), Tailwind CSS (Dark Mode supported), Lucide Icons
*   **Database:** SQLite (Embedded relational database (Production path: PostgreSQL))

## Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/h-rishi16/sentinel.git
   cd sentinel
   ```

2. **Set up a virtual environment and install dependencies:**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -e .
   ```

3. **Seed the Synthetic Database:**
   Ensure the database is populated with financial data for the engines to run:
   ```bash
   python scripts/seed_db.py 1000 5000 20000 0.002 0.0 normal normal
   ```
   *(Or simply run the application and use the **Data Forge** UI to generate a dataset).*

## Running the Application

Start the FastAPI ASGI server using Uvicorn:

```bash
uvicorn sentinel.api.main:app --host 0.0.0.0 --port 8000 --reload
```

Then open your browser and navigate to: [Sentinel](https://sentinel-risk-platform.onrender.com/)

## License

This project is open-source and available under the [MIT License](LICENSE).
