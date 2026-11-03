# Basic House Price Prediction

A small Python machine-learning project that predicts an advertised house price from its carpet area using linear regression.

## About

The script reads `house_prices.csv`, cleans the carpet-area and price columns, converts prices into lakh rupees, fits a linear regression model, and estimates the price of a 600 sq. ft. property. The dataset includes prices expressed in lakh and crore; rows with missing or unsupported values are excluded.

## Project files

- `Basic_House_Price_Prediction.py` — data preparation, model training, prediction, and chart generation.
- `house_prices.csv` — input data.
- `output.txt` — prediction and number of training properties used.
- `house_price_prediction.png` — scatter plot and fitted regression line.

The output files are generated or updated whenever the script runs.

## Requirements

- Python 3.9 or later
- pandas
- matplotlib
- scikit-learn

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

## Run

```bash
python Basic_House_Price_Prediction.py
```

The script writes `output.txt` and `house_price_prediction.png` beside the Python file, prints the prediction in the terminal, and opens the chart in an interactive window.

## Limitations

This is an educational example using a small dataset and a single feature. Its prediction is not a property valuation and should not be used for financial decisions.
