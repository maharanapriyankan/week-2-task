
"""
WEEK 2 PROJECT
Title: Data Cleaning and Preprocessing for Logistics Analysis
Dataset: Brazilian E-Commerce Public Dataset by Olist
Scenario: BiharKart Logistics Analysis

Operations:
1. Load datasets
2. Check missing values
3. Handle missing values
4. Remove duplicate records
5. Convert date columns
6. Detect and handle outliers
7. Feature engineering
8. Merge datasets
9. Scale numerical data
10. Export cleaned dataset
"""

# =====================================================
# 1. IMPORT LIBRARIES
# =====================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import os

print("=" * 60)
print("WEEK 2: LOGISTICS DATA CLEANING AND PREPROCESSING")
print("=" * 60)


# =====================================================
# 2. LOAD DATASETS
# =====================================================

DATA_FOLDER = "data"

try:
    customers = pd.read_csv(
        os.path.join(DATA_FOLDER, "olist_customers_dataset.csv")
    )

    orders = pd.read_csv(
        os.path.join(DATA_FOLDER, "olist_orders_dataset.csv")
    )

    items = pd.read_csv(
        os.path.join(DATA_FOLDER, "olist_order_items_dataset.csv")
    )

    payments = pd.read_csv(
        os.path.join(DATA_FOLDER, "olist_order_payments_dataset.csv")
    )

    products = pd.read_csv(
        os.path.join(DATA_FOLDER, "olist_products_dataset.csv")
    )

except FileNotFoundError as error:
    print("Dataset file not found:", error)
    print("Place all Olist CSV files inside the 'data' folder.")
    raise SystemExit(1)

print("\nDatasets loaded successfully!")

print("\nCustomers:", customers.shape)
print("Orders:", orders.shape)
print("Items:", items.shape)
print("Payments:", payments.shape)
print("Products:", products.shape)


# =====================================================
# 3. CHECK MISSING VALUES
# =====================================================

print("\n--- MISSING VALUE ANALYSIS ---")

for name, df in {
    "Customers": customers,
    "Orders": orders,
    "Items": items,
    "Payments": payments,
    "Products": products
}.items():

    print(f"\n{name} missing values:")
    print(df.isnull().sum())


# =====================================================
# 4. HANDLE MISSING VALUES
# =====================================================

# Fill missing product category
products["product_category_name"] = (
    products["product_category_name"].fillna("Unknown")
)

# Fill missing product dimensions with median
dimension_columns = [
    "product_weight_g",
    "product_length_cm",
    "product_height_cm",
    "product_width_cm"
]

for col in dimension_columns:
    if products[col].notna().any():
        products[col] = products[col].fillna(
            products[col].median()
        )

# Remove records without purchase timestamp
orders = orders.dropna(
    subset=["order_purchase_timestamp"]
)

print("\nMissing values handled successfully!")


# =====================================================
# 5. REMOVE DUPLICATE RECORDS
# =====================================================

print("\n--- DUPLICATE ANALYSIS ---")

for name, df in {
    "Customers": customers,
    "Orders": orders,
    "Items": items,
    "Payments": payments,
    "Products": products
}.items():

    print(f"{name} duplicates:", df.duplicated().sum())

customers = customers.drop_duplicates()
orders = orders.drop_duplicates()
items = items.drop_duplicates()
payments = payments.drop_duplicates()
products = products.drop_duplicates()

print("\nDuplicate records removed!")


# =====================================================
# 6. CONVERT DATE COLUMNS
# =====================================================

date_columns = [
    "order_purchase_timestamp",
    "order_approved_at",
    "order_delivered_carrier_date",
    "order_delivered_customer_date",
    "order_estimated_delivery_date"
]

for col in date_columns:
    orders[col] = pd.to_datetime(
        orders[col],
        errors="coerce"
    )

print("\nDate columns converted successfully!")


# =====================================================
# 7. OUTLIER DETECTION
# =====================================================

print("\n--- OUTLIER DETECTION ---")

Q1 = items["price"].quantile(0.25)
Q3 = items["price"].quantile(0.75)

IQR = Q3 - Q1

lower_limit = max(0, Q1 - 1.5 * IQR)
upper_limit = Q3 + 1.5 * IQR

print("Q1:", Q1)
print("Q3:", Q3)
print("IQR:", IQR)
print("Lower limit:", lower_limit)
print("Upper limit:", upper_limit)

outliers = items[
    (items["price"] < lower_limit) |
    (items["price"] > upper_limit)
]

print("Number of price outliers:", len(outliers))


# =====================================================
# 8. HANDLE OUTLIERS
# =====================================================

items["price_cleaned"] = items["price"].clip(
    lower=lower_limit,
    upper=upper_limit
)

print("\nOutliers treated using IQR-based clipping!")


# =====================================================
# 9. FEATURE ENGINEERING
# =====================================================

orders["delivery_time_days"] = (
    orders["order_delivered_customer_date"] -
    orders["order_purchase_timestamp"]
).dt.total_seconds() / (24 * 60 * 60)

print("\nDelivery time calculated!")

print(
    orders[
        ["order_id", "delivery_time_days"]
    ].head()
)


# =====================================================
# 10. MERGE DATASETS
# =====================================================

# Merge orders and order items
logistics_data = orders.merge(
    items[
        ["order_id", "product_id", "price", "price_cleaned",
         "freight_value"]
    ],
    on="order_id",
    how="left"
)

# Merge customer details
logistics_data = logistics_data.merge(
    customers[
        ["customer_id", "customer_city",
         "customer_state"]
    ],
    on="customer_id",
    how="left"
)

# Merge product details
logistics_data = logistics_data.merge(
    products[
        ["product_id", "product_category_name",
         "product_weight_g"]
    ],
    on="product_id",
    how="left"
)

# Merge payment details
payment_summary = payments.groupby(
    "order_id",
    as_index=False
).agg(
    payment_value=("payment_value", "sum"),
    payment_installments=("payment_installments", "max")
)

logistics_data = logistics_data.merge(
    payment_summary,
    on="order_id",
    how="left"
)

print("\nDatasets merged successfully!")
print("Final dataset shape:", logistics_data.shape)


# =====================================================
# 11. DATA SCALING
# =====================================================

scaler = MinMaxScaler()

logistics_data["price_scaled"] = scaler.fit_transform(
    logistics_data[["price_cleaned"]].fillna(
        logistics_data["price_cleaned"].median()
    )
)

print("\nPrice scaling completed!")

print(
    logistics_data[
        ["price", "price_cleaned", "price_scaled"]
    ].head()
)


# =====================================================
# 12. BASIC LOGISTICS ANALYSIS
# =====================================================

print("\n--- LOGISTICS ANALYSIS ---")

print(
    "\nAverage delivery time:",
    round(orders["delivery_time_days"].mean(), 2),
    "days"
)

print(
    "\nTotal number of orders:",
    orders["order_id"].nunique()
)

print(
    "\nTotal logistics records:",
    len(logistics_data)
)


# =====================================================
# 13. DATA VISUALIZATION
# =====================================================

plt.figure(figsize=(8, 5))

sns.boxplot(x=items["price"])

plt.title("Product Price Before Outlier Treatment")
plt.xlabel("Price")

plt.tight_layout()
plt.savefig("price_before_cleaning.png")
plt.show()


plt.figure(figsize=(8, 5))

sns.boxplot(x=items["price_cleaned"])

plt.title("Product Price After Outlier Treatment")
plt.xlabel("Cleaned Price")

plt.tight_layout()
plt.savefig("price_after_cleaning.png")
plt.show()


# =====================================================
# 14. SAVE CLEANED DATASET
# =====================================================

OUTPUT_FILE = "week2_cleaned_logistics_data.csv"

logistics_data.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\nCleaned dataset saved as:", OUTPUT_FILE)


# =====================================================
# 15. FINAL REPORT
# =====================================================

print("\n" + "=" * 60)
print("WEEK 2 PROJECT COMPLETED SUCCESSFULLY")
print("=" * 60)

print("Customers:", customers.shape)
print("Orders:", orders.shape)
print("Items:", items.shape)
print("Products:", products.shape)

print("\nFinal logistics dataset:", logistics_data.shape)

print("\nCompleted tasks:")
print("1. Dataset loading")
print("2. Missing value handling")
print("3. Duplicate removal")
print("4. Date conversion")
print("5. Outlier detection and treatment")
print("6. Feature engineering")
print("7. Dataset merging")
print("8. Numerical data scaling")
print("9. Data visualization")
print("10. Exporting cleaned dataset")

print("\nWEEK 2 PREPROCESSING FINISHED!")
