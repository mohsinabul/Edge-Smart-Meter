# Edge-Smart-Meter

Forecasting Household Smart Meter Energy with Lightweight Transformers

# Overview

This project explores how modern machine learning models can be applied to forecast household electricity consumption using smart meter data.

We designed, trained, and evaluated four models:

LSTM - a recurrent baseline for sequential data

TinyFormer - lightweight Transformer with patching

LiPFormer - linear-attention Transformer

LiteFormer - efficient hybrid Transformer

The work was done in two stages:

V1 → Forecasts using only raw consumption history (lags).

V2 → Extended with contextual features (lags, rolling means, calendar flags such as weekends and holidays).

Finally, we went beyond static evaluation and deployed the best model in a live mimic environment with Docker + Streamlit, running on a small Azure VM, simulating how a smart meter could forecast the next 6 hours in real-time.

# Research Objective

To evaluate whether lightweight Transformers can rival or surpass LSTM models in forecasting household energy consumption, while remaining explainable and feasible for edge deployment.

# Key Findings

Adding contextual features in V2 boosted accuracy across all models.

TinyFormer and LiPFormer showed the biggest R² gains (over 100%).

LiteFormer balanced smooth forecasts with efficiency.

LSTM improved only modestly, confirming the advantage of contextual Transformers.

SHAP analysis confirmed that V1 models leaned on recent lags only, while V2 models learned to use rolling means, calendar flags, and holiday effects.

On the edge test, all models met the 1s p95 latency requirement. TinyFormer and LiteFormer were the most efficient, while LiPFormer provided the strongest accuracy.

# Installation & Usage
1. Clone the repo
git clone https://github.com/mohsinabul/Edge-Smart-Meter.git
cd EdgeMeter_AI

2. Local training environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.dev.txt

3. Run evaluation

Example: Extra Trees or Transformer training scripts inside src/.

python src/01_train_LSTM.py
python src/01_train_TinyFormer.py


Outputs are saved into outputs/ with metrics, plots, and logs.

# Live Smart Meter Mimic

The highlight of this project is a real-time mimic that continuously predicts household demand.

It runs two Docker services:

Producer — reads demo arrays every 30 minutes, uses a TFLite model to predict the next 6 hours, and writes results to a log.

Dashboard — Streamlit app that visualises the live log with KPIs, forecast plots, and cumulative usage.

Run locally
cd smart_meter_mimic
docker build -t edgemeter:latest .
docker compose up -d


Live Dashboard: https://edgemeter.duckdns.org

Run on Azure VM

VM size: Standard_B1ms (1 vCPU / 2 GiB RAM, Ubuntu 22.04)

Docker + Docker Compose installed

Model and demo arrays copied into ~/EdgeMeter_AI/smart_meter_mimic/data/

Service auto-starts on reboot via systemd

Optional HTTPS configured via DuckDNS + Let’s Encrypt

# Requirements

We keep two dependency sets:

requirements.dev.txt → for local training, SHAP, and evaluation

requirements.runtime.txt → minimal runtime stack for Docker mimic

Pinned versions are used to avoid NumPy/TFLite ABI conflicts.

# Explainability

SHAP was applied to both V1 and V2.

V1: models rely almost entirely on t-0, t-1, t-2 lags.

V2: importance spreads to rolling averages, weekend flags, and holidays.

This matches the performance boost in winter/holiday peaks.

# Novelty

Unlike most studies that stop at offline accuracy, this project demonstrated a full pipeline:

Feature engineering that materially improved forecasts.

Lightweight Transformers that rivalled and often beat LSTM.

Explainability with SHAP to ensure transparency.

Edge-ready deployment with Docker, Azure VM, and a live dashboard.

This end-to-end approach bridges academic modelling and practical utility for real smart meters.

# Acknowledgements

This work was completed as part of the MSc Data Science at Atlantic Technological University, Donegal.
Thanks to my supervisors for their guidance, and to friends and colleagues for support during the project
