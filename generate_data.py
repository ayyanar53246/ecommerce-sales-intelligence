
from pathlib import Path
import sqlite3
import numpy as np
import pandas as pd

# Create data folder
Path("data").mkdir(exist_ok=True)

np.random.seed(42)

# Product master table
products = pd.DataFrame({
    "product_id": range(101, 111),
    "product_name": [
        "Laptop", "Mouse", "Keyboard", "Headphones",
        "Smartphone", "Monitor", "USB Cable", "Webcam",
        "Power Bank", "Tablet"
    ],
    "category": [
        "Electronics", "Accessories", "Accessories",
        "Accessories", "Electronics", "Electronics",
        "Accessories", "Electronics", "Accessories",
        "Electronics"
    ],
    "unit_price": [
        55000, 800, 1500, 2200, 30000,
        12000, 300, 1800, 1600, 22000
    ],
    "unit_cost": [
        44000, 450, 900, 1300, 24000,
        9000, 120, 1100, 950, 17500
    ],
    "current_stock": [
        8, 250, 120, 15, 12, 35, 500, 18, 70, 10
    ],
    "reorder_level": [
        10, 50, 30, 20, 15, 10, 100, 15, 25, 12
    ]
})

# Create 500 sample orders
n = 500
idx = np.random.randint(0, len(products), n)
selected = products.iloc[idx].reset_index(drop=True)

orders = pd.DataFrame({
    "order_id": [f"ORD{i:04d}" for i in range(1, n + 1)],
    "order_date": pd.to_datetime(
        np.random.choice(
            pd.date_range("2025-01-01", "2026-10-25"),
            n
        )
    ),
    "product_id": selected["product_id"],
    "quantity": np.random.randint(1, 5, n),
    "discount": np.random.choice([0, 0.05, 0.10, 0.15], n),
    "region": np.random.choice(
        ["Chennai", "Bengaluru", "Mumbai", "Delhi", "Hyderabad"],
        n
    )
})

orders["unit_price"] = selected["unit_price"].to_numpy()
orders["unit_cost"] = selected["unit_cost"].to_numpy()

orders["revenue"] = (
    orders["quantity"] * orders["unit_price"]
    * (1 - orders["discount"])
).round(2)

orders["profit"] = (
    orders["revenue"]
    - orders["quantity"] * orders["unit_cost"]
).round(2)

# Generate illustrative return records
returns = orders.sample(frac=0.08, random_state=42)[
    ["order_id", "product_id"]
].copy()

returns["return_reason"] = np.random.choice(
    ["Damaged", "Wrong item", "Not as expected", "Other"],
    len(returns)
)

# Save CSV files
products.to_csv("data/products.csv", index=False)
orders.to_csv("data/orders.csv", index=False)
returns.to_csv("data/returns.csv", index=False)

# Create SQLite database
with sqlite3.connect("data/ecommerce.db") as conn:
    products.to_sql("products", conn, if_exists="replace", index=False)
    orders.to_sql("orders", conn, if_exists="replace", index=False)
    returns.to_sql("returns", conn, if_exists="replace", index=False)

print("Dataset and SQLite database created successfully!")
print("Products:", len(products))
print("Orders:", len(orders))
print("Illustrative returns:", len(returns))
print(f"Revenue: Rs. {orders['revenue'].sum():,.2f}")
print(f"Profit: Rs. {orders['profit'].sum():,.2f}")
print("Database: data/ecommerce.db")
