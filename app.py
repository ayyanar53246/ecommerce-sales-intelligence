
import sqlite3
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(
    page_title="E-Commerce Sales Intelligence",
    page_icon="📊",
    layout="wide"
)

st.title("E-Commerce Sales Intelligence Dashboard")
st.caption("Product profitability | Inventory risk | Returns analysis")

# Load data from SQLite using SQL
@st.cache_data
def load_data():
    with sqlite3.connect("data/ecommerce.db") as conn:
        orders = pd.read_sql_query(
            """
            SELECT o.*, p.product_name, p.category,
                   p.current_stock, p.reorder_level
            FROM orders o
            JOIN products p ON o.product_id = p.product_id
            """,
            conn,
            parse_dates=["order_date"]
        )

        products = pd.read_sql_query(
            "SELECT * FROM products", conn
        )

        returns = pd.read_sql_query(
            "SELECT * FROM returns", conn
        )

    return orders, products, returns


orders, products, returns = load_data()

# Sidebar filters
st.sidebar.header("Filters")

categories = ["All"] + sorted(orders["category"].unique().tolist())
category = st.sidebar.selectbox("Product category", categories)

start_date = st.sidebar.date_input(
    "Start date", orders["order_date"].min().date()
)
end_date = st.sidebar.date_input(
    "End date", orders["order_date"].max().date()
)

filtered = orders[
    (orders["order_date"].dt.date >= start_date) &
    (orders["order_date"].dt.date <= end_date)
].copy()

if category != "All":
    filtered = filtered[filtered["category"] == category]

if start_date > end_date:
    st.error("Start date must be before end date.")
    st.stop()

if filtered.empty:
    st.warning("No sales records found for these filters.")
    st.stop()

# KPI calculations using NumPy
revenue = filtered["revenue"].sum()
profit = filtered["profit"].sum()
order_count = filtered["order_id"].nunique()
aov = revenue / order_count if order_count else 0
margin = np.where(revenue != 0, profit / revenue * 100, 0)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Revenue", f"₹{revenue:,.0f}")
c2.metric("Total Profit", f"₹{profit:,.0f}")
c3.metric("Orders", f"{order_count:,}")
c4.metric("Profit Margin", f"{float(margin):.1f}%")

st.divider()

# Monthly revenue trend
st.subheader("1. Monthly Revenue Trend")

monthly = (
    filtered.set_index("order_date")
    .resample("MS")["revenue"]
    .sum()
    .reset_index()
)

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(monthly["order_date"], monthly["revenue"], marker="o")
ax.set_xlabel("Month")
ax.set_ylabel("Revenue (₹)")
ax.grid(alpha=0.3)
fig.tight_layout()
st.pyplot(fig)
plt.close(fig)

# Product profitability analysis
st.subheader("2. Product Performance")

performance = (
    filtered.groupby("product_name")
    .agg(
        Revenue=("revenue", "sum"),
        Profit=("profit", "sum"),
        Units_Sold=("quantity", "sum")
    )
    .sort_values("Profit", ascending=False)
)

fig, ax = plt.subplots(figsize=(10, 5))
performance["Profit"].plot(kind="bar", ax=ax)
ax.set_ylabel("Profit (₹)")
ax.set_xlabel("Product")
ax.tick_params(axis="x", rotation=35)
fig.tight_layout()
st.pyplot(fig)
plt.close(fig)

st.dataframe(performance, use_container_width=True)

# Slow-moving product analysis
st.subheader("3. Slow-Moving Product Analysis")

product_sales = (
    filtered.groupby(
        ["product_id", "product_name", "category"]
    )
    .agg(
        Units_Sold=("quantity", "sum"),
        Revenue=("revenue", "sum"),
        Profit=("profit", "sum")
    )
    .reset_index()
)

# Include products with zero sales in the selected period
product_sales = products[
    ["product_id", "product_name", "category"]
].merge(
    product_sales,
    on=["product_id", "product_name", "category"],
    how="left"
)

product_sales[["Units_Sold", "Revenue", "Profit"]] = (
    product_sales[["Units_Sold", "Revenue", "Profit"]].fillna(0)
)

# Use the median as a simple comparison benchmark
sales_median = product_sales["Units_Sold"].median()

product_sales["Sales_Status"] = np.where(
    product_sales["Units_Sold"] < sales_median,
    "Review Sales",
    "Normal"
)

slow_products = product_sales[
    product_sales["Sales_Status"] == "Review Sales"
].sort_values("Units_Sold")

if slow_products.empty:
    st.info("No products fall below the selected-period median.")
else:
    st.write(
        "Products below the median units sold in the selected period. "
        "This is a screening indicator, not proof of poor demand."
    )
    st.dataframe(
        slow_products[
            ["product_name", "Units_Sold", "Revenue", "Profit", "Sales_Status"]
        ],
        use_container_width=True
    )

csv_data = orders.to_csv(index=False).encode("utf-8")

st.download_button(
    label="Download Sales Report (CSV)",
    data=csv_data,
    file_name="sales_report.csv",
    mime="text/csv"
)

# Inventory risk
st.subheader("3. Inventory Reorder Alerts")

inventory = products.copy()
inventory["stock_status"] = np.where(
    inventory["current_stock"] <= inventory["reorder_level"],
    "Reorder",
    "Stock OK"
)

alerts = inventory[inventory["stock_status"] == "Reorder"]

if alerts.empty:
    st.success("No products currently meet the reorder alert rule.")
else:
    st.warning(f"{len(alerts)} products need inventory review.")
    st.dataframe(
        alerts[
            ["product_name", "current_stock", "reorder_level", "stock_status"]
        ],
        use_container_width=True
    )

# Returns analysis
st.subheader("4. Product Returns")

return_summary = (
    returns.groupby("product_id")
    .agg(Returns=("order_id", "nunique"))
    .reset_index()
)

sold_summary = (
    orders.groupby("product_id")
    .agg(Orders=("order_id", "nunique"))
    .reset_index()
)

return_summary = return_summary.merge(
    sold_summary, on="product_id", how="right"
).fillna({"Returns": 0})

return_summary["Return Rate (%)"] = np.where(
    return_summary["Orders"] > 0,
    return_summary["Returns"] / return_summary["Orders"] * 100,
    0
)

return_summary = return_summary.merge(
    products[["product_id", "product_name"]],
    on="product_id",
    how="left"
)

st.dataframe(
    return_summary.sort_values("Return Rate (%)", ascending=False),
    use_container_width=True
)

# Correlation heatmap
st.subheader("5. Sales Data Relationship")

fig, ax = plt.subplots(figsize=(7, 4))
sns.heatmap(
    filtered[["quantity", "unit_price", "discount", "revenue", "profit"]]
    .corr(),
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
    ax=ax
)
fig.tight_layout()
st.pyplot(fig)
plt.close(fig)

# Business insights
st.subheader("6. Business Recommendations")


# CSV Report Download
st.subheader("7. Download Sales Report")

csv_data = filtered.to_csv(index=False).encode("utf-8")

st.download_button(
    label="📥 Download Sales Report (CSV)",
    data=csv_data,
    file_name="ecommerce_sales_report.csv",
    mime="text/csv"
)


top_product = performance["Profit"].idxmax()
low_product = performance["Profit"].idxmin()

st.write(
    f"- **Top profit contributor:** {top_product}. "
    "Review its demand and maintain appropriate stock."
)
st.write(
    f"- **Lowest total profit in this selection:** {low_product}. "
    "Investigate sales volume, cost, discounts and pricing before taking action."
)
st.write(
    "- **Inventory:** Review products flagged for reorder against actual demand "
    "and supplier lead times before replenishing."
)
st.write(
    "- **Returns:** Investigate return reasons and sample-data limitations "
    "before making product quality decisions."
)

st.caption(
    "Portfolio demo using synthetic sales data. "
    "Return records and stock levels are illustrative, not live company data."
)
