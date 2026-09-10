# Renewable Grid Operations Agent

A beginner-friendly Streamlit application demonstrating deterministic power-system analysis, Generative AI, and agentic AI using Groq.

## Files

- `app.py` — complete application
- `requirements.txt` — Python dependencies
- `grid_input_24h.csv` — fixed 24-hour input dataset

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Set your Groq API key as `GROQ_API_KEY` in Streamlit secrets.

## Streamlit Cloud

Upload these files to a GitHub repository, deploy `app.py` as the main file, and add:

```toml
GROQ_API_KEY = "your-key-here"
```

to the app's Secrets settings.

## Engineering concept

The application calculates:

- renewable generation
- net load
- battery charging/discharging and SOC
- grid import
- renewable curtailment
- critical deficit conditions

The Groq model can call four simple engineering tools and then generate an explanation and operational recommendations.

This is an educational decision-support prototype, not a real-time grid controller.
