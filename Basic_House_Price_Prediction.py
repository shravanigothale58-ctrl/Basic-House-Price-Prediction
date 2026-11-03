
# Basic House Price Prediction Mini Project
# Uses only Carpet Area and Amount(in rupees) from house_prices.csv.

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from pathlib import Path

# Step 1: Read the CSV file (keep this file in the same folder)
project_dir = Path(__file__).resolve().parent
df = pd.read_csv(project_dir / "house_prices.csv")

# Step 2: Keep only the two columns needed for this mini project
df = df[["Carpet Area", "Amount(in rupees)"]].copy()

# Step 3: Convert area text such as "500 sqft" into a number
# Extract the first number from the Carpet Area column.
df["Carpet Area"] = df["Carpet Area"].astype(str).str.extract(r"(\d+(?:\.\d+)?)")[0]
df["Carpet Area"] = pd.to_numeric(df["Carpet Area"], errors="coerce")

# Step 4: Convert advertised prices such as "42 Lac" and "1.40 Cr" into lakh rupees.
def price_in_lakhs(price):
    price = str(price).strip().lower().replace(",", "")

    if "call for price" in price or price == "nan":
        return None

    # Crore: 1 Cr = 100 lakh
    if "cr" in price or "crore" in price:
        number = pd.to_numeric(pd.Series([price]).str.extract(r"(\d+(?:\.\d+)?)")[0], errors="coerce").iloc[0]
        return number * 100 if pd.notna(number) else None

    # Lakh: keep the number as it is
    if "lac" in price or "lakh" in price:
        number = pd.to_numeric(pd.Series([price]).str.extract(r"(\d+(?:\.\d+)?)")[0], errors="coerce").iloc[0]
        return number if pd.notna(number) else None

    # Ignore prices that do not match these simple formats.
    return None

df["Price_Lakhs"] = df["Amount(in rupees)"].apply(price_in_lakhs)

# Step 5: Remove rows where area or price is missing or invalid
df = df.dropna(subset=["Carpet Area", "Price_Lakhs"])

# Keep realistic positive values for this simple example
df = df[(df["Carpet Area"] > 0) & (df["Price_Lakhs"] > 0)]

# Step 6: Select input (X) and output (y)
X = df[["Carpet Area"]]  # Input: carpet area in square feet
y = df["Price_Lakhs"]    # Output: price in lakh rupees

# Step 7: Train a simple Linear Regression model
model = LinearRegression()
model.fit(X, y)

# Step 8: Predict price for a 600-square-foot property
new_house = pd.DataFrame({"Carpet Area": [600]})
predicted_price = model.predict(new_house)[0]
prediction = round(predicted_price, 2)
prediction_text = (
    f"Predicted price for a 600 sqft property: {prediction} lakh rupees\n"
    f"Training properties used: {len(df)}\n"
)
(project_dir / "output.txt").write_text(prediction_text, encoding="utf-8")
print(prediction_text.strip())

# Step 9: Draw a simple graph
plt.scatter(X["Carpet Area"], y, alpha=0.25, label="CSV properties")
plt.plot(X["Carpet Area"], model.predict(X), color="red", label="Prediction line")
plt.xlabel("Carpet Area (sq. ft.)")
plt.ylabel("Advertised Price (lakh rupees)")
plt.title("Basic House Price Prediction")
plt.legend()
plt.tight_layout()
plt.savefig(project_dir / "house_price_prediction.png", dpi=150)
plt.show()
