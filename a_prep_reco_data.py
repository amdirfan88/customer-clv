import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
#from tqdm import tqdm
#import re
import os
import pickle
#from tqdm.notebook import tqdm
from tqdm.auto import tqdm
from collections import defaultdict
from urllib.request import urlretrieve

print('Current working directory is', os.getcwd())

# downloading and combining data from excel sheets
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00502/online_retail_II.xlsx"
if not os.path.exists("online_retail_II.xlsx"):
    with tqdm(desc="Downloading online_retail_II.xlsx", unit="B", unit_scale=True) as download_progress:
        urlretrieve(
            url,
            "online_retail_II.xlsx",
            reporthook=lambda blocks, block_size, total_size: (
                setattr(download_progress, 'total', total_size if total_size >= 0 else None),
                download_progress.update(max(0, (min(blocks * block_size, total_size) if total_size >= 0 else blocks * block_size) - download_progress.n)),
            ),
        )

with tqdm(total=1, desc='Reading online_retail_II.xlsx (Year 2009-2010)', unit="task", leave=False) as task_progress:
    df_2010 = pd.read_excel("online_retail_II.xlsx", sheet_name="Year 2009-2010")
    task_progress.update(1)
with tqdm(total=1, desc='Reading online_retail_II.xlsx (Year 2010-2011)', unit="task", leave=False) as task_progress:
    df_2011 = pd.read_excel("online_retail_II.xlsx", sheet_name="Year 2010-2011")
    task_progress.update(1)

# Check if the file exists before trying to delete it
filename = "online_retail_II.xlsx"
if os.path.exists(filename):
    os.remove(filename)
    print(f"Successfully deleted {filename}")
else:
    print(f"The file {filename} does not exist.")

df_total = pd.concat([df_2010, df_2011], ignore_index=True)

#df_total.to_csv("data/downloaded_data/online_retail_raw.csv", index=False)

# reading previously saved csv, which contains combined data of two excel sheets
#df_total = pd.read_csv("data/downloaded_data/online_retail_raw.csv")

df_total.info()

df_total.head(4)

df_total['InvoiceDate'] = pd.to_datetime(df_total['InvoiceDate'])
df_total = df_total.rename(columns={"Customer ID": "Customer_ID"})

# Split date
split_date = pd.Timestamp("2010-12-05")

# Saving to CSV calibration datasets
df_calib = df_total[df_total["InvoiceDate"] <= split_date].copy()
if not os.path.exists('data/given_data/calibration_raw.csv'):
    with tqdm(total=1, desc='Saving calibration_raw.csv', unit="task", leave=False) as task_progress:
        df_calib.to_csv('data/given_data/calibration_raw.csv', index=False)
        task_progress.update(1)

# Holdout datasets
df = df_total[df_total["InvoiceDate"] > split_date].copy()

print(f"Calibration transactions: {len(df_calib)}")
print(f"Holdout transactions    : {len(df)}")

# Deleting lines that put as a test, or as fee	
df_calib = df_calib[~df_calib["Description"].isin(["CHECKED", "CHECK", "CHECK?", "STOCK CHECK", "CARRIAGE"])]
df_calib = df_calib[~df_calib["Description"].str.contains("test|fee|damaged|BANK CHARGES|DISCOUNT|POSTAGE|MANUAL|ADJUST|test|lost|samples", case=False, na=False)]

# 5. Remove administrative/service stock codes
non_product_codes = ['DOT', 'POST', 'C2', 'M', 'BANK CHARGES', 'PADS', 'CRUK']
df_calib = df_calib[~df_calib['StockCode'].isin(non_product_codes)]

with tqdm(total=1, desc='Calculating product price statistics', unit="task", leave=False) as task_progress:
    tmp = df_calib.groupby("StockCode").agg(minprice=("Price","min"), maxprice=("Price","max"), avgprice=("Price","mean"), std_of_price=("Price","std"))
    task_progress.update(1)
with tqdm(total=1, desc='Selecting product descriptions', unit="task", leave=False) as task_progress:
    desc = df_calib.groupby(["StockCode","Description"]).size().reset_index(name="n").sort_values(["StockCode","n"], ascending=[True,False]).drop_duplicates("StockCode").set_index("StockCode")[["Description"]]
    task_progress.update(1)
df_calib_product = desc.join(tmp).reset_index()
df_calib_product["std_of_price"] = df_calib_product["std_of_price"].fillna(0)
df_calib_product.head()

#df_calib.drop(columns='Description', inplace=True)

if not os.path.exists('data/given_data/prod_catalog_calib.csv'):
    with tqdm(total=1, desc='Saving prod_catalog_calib.csv', unit="task", leave=False) as task_progress:
        df_calib_product.to_csv('data/given_data/prod_catalog_calib.csv', index=False)
        task_progress.update(1)
if not os.path.exists('data/given_data/prod_catalog_calib.parquet'):
    with tqdm(total=1, desc='Saving prod_catalog_calib.parquet', unit="task", leave=False) as task_progress:
        df_calib_product.astype({'StockCode': 'string'}).to_parquet('data/given_data/prod_catalog_calib.parquet', index=False)
        task_progress.update(1)

df['Invoice'] = df['Invoice'].astype(str).str.upper().str.strip()
df["Customer_ID"] = df["Customer_ID"].astype("Int64")
df['StockCode'] = df['StockCode'].astype(str).str.upper().str.strip()
df['Description'] = df['Description'].astype(str).str.upper().str.strip()
df['Country'] = df['Country'].astype(str).str.upper().str.strip()
df.info()

# Both negative price and negative quantity mean cancellation of orders; 
# I standardize cancellation of order by considering Quantity -ve and price positive
df['Quantity'] = df['Quantity'] * np.sign(df['Price'])
df['Price'] = np.abs(df['Price'])

# Sort by time
df = df.sort_values("InvoiceDate").reset_index(drop=True)

# Status column
df["status"] = pd.NA

# ---------------------------------------------------
# Matching structure:
# key = (Customer_ID, StockCode, abs(Quantity), Price)
# value = list of unmatched positive row positions
# ---------------------------------------------------
open_sales = defaultdict(list)

# Use tqdm for progress tracking
for i, row in tqdm(
    enumerate(df.itertuples(index=False)),
    total=len(df),
    desc="Matching returns"
):
    key = (
        row.Customer_ID,
        row.StockCode,
        abs(row.Quantity),
        row.Price
    )
    if row.Quantity > 0:
        open_sales[key].append(i)

    elif row.Quantity < 0:
        if open_sales[key]:
            match_i = open_sales[key].pop()
            df.at[match_i, "status"] = "cancelled"
            df.at[i, "status"] = "return: purchase found"
        else:
            df.at[i, "status"] = "return: purchase not found"

# Any remaining positive rows are completed purchases
df.loc[(df["Quantity"] > 0) & (df["status"].isna()), "status"] = "completed"

df['status'] = df['status'].astype(str)

## Keeping only completed orders
df = df[df['status']=='completed']
#df_return = df[df['status'] == 'return: purchase found']

# droping status column before saving to csv
df.drop(columns = 'status', inplace=True)
#df_return.drop(columns = 'status', inplace=True)

# Deleting lines that put as a test, or as fee	
df = df[~df["Description"].isin(["CHECKED", "CHECK", "CHECK?", "STOCK CHECK", "CARRIAGE"])]
df = df[~df["Description"].str.contains("test|fee|damaged|BANK CHARGES|DISCOUNT|POSTAGE|MANUAL|ADJUST|test|lost|samples", case=False, na=False)]

# 5. Remove administrative/service stock codes
non_product_codes = ['DOT', 'POST', 'C2', 'M', 'BANK CHARGES', 'PADS', 'CRUK']
df = df[~df['StockCode'].isin(non_product_codes)]

### Checking whether rows corresponds to a same invoice referes to a same Customer_ID 

# Count unique Customer_ID  per Invoice
# dropna=False means missing Customer_ID is counted as its own value
with tqdm(total=1, desc='Checking invoice customers', unit="task", leave=False) as task_progress:
    invoice_check = (
        df.groupby('Invoice')
          .agg(
              n_customers=('Customer_ID', lambda x: x.nunique(dropna=False))
          )
          .reset_index()
    )
    task_progress.update(1)

# Invoices where same invoice has different customer 
bad_invoices = invoice_check[(invoice_check['n_customers'] > 1)]

# Save all original rows belonging to those problematic invoices
invoice_exceptions = (
    df[df['Invoice'].isin(bad_invoices['Invoice'])]
    .sort_values(['Invoice', 'Customer_ID'])
)


if bad_invoices.shape[0] == 0:
    print('No mismatch found between invoice and Customer_ID.  ✅')
else:
    print(f'{bad_invoices.shape[0]} mismatches observed between invoice and Customer_ID. Some mismatches as examples are:')
    print(invoice_exceptions[['Invoice', 'Customer_ID', 'Country']].drop_duplicates().head(5))

# Number of duplicate rows (considering all columns)
n_duplicates = df.duplicated().sum()

if n_duplicates == 0:
    print("✅ No duplicate rows found.")
else:
    print(f"❌ Found {n_duplicates:,} duplicate rows.")

    # Drop duplicates
    df = df.drop_duplicates().reset_index(drop=True)

    print(f"Deleted {n_duplicates:,} duplicate rows.")
    print(f"Remaining rows: {len(df):,}")

# Standardarizing the product description
with tqdm(total=1, desc='Standardizing product descriptions', unit="task", leave=False) as task_progress:
    collapsed = (
        df.groupby(['StockCode', 'Description'])
          .size()
          .rename('count')
          .reset_index()
    )
    task_progress.update(1)

with tqdm(total=1, desc='Grouping data', unit="task", leave=False) as task_progress:
    collapsed['rank'] = (
        collapsed.groupby('StockCode')['count']
                 .rank(method='first', ascending=False)
    )
    task_progress.update(1)

collapsed = collapsed[collapsed['rank'] == 1]

df = df.merge(collapsed, on='StockCode', how='left', suffixes=('', '_top'))
df['Description'] = df['Description_top']
df = df.drop(columns=['Description_top', 'rank', 'count'])

if not os.path.exists('data/given_data/holdout_transaction.csv'):
    with tqdm(total=1, desc='Saving holdout_transaction.csv', unit="task", leave=False) as task_progress:
        df.to_csv('data/given_data/holdout_transaction.csv', index=False)
        task_progress.update(1)

with tqdm(total=1, desc='Calculating product price statistics', unit="task", leave=False) as task_progress:
    tmp = df.groupby("StockCode").agg(minprice=("Price","min"), maxprice=("Price","max"), avgprice=("Price","mean"), std_of_price=("Price","std"))
    task_progress.update(1)
with tqdm(total=1, desc='Selecting product descriptions', unit="task", leave=False) as task_progress:
    desc = df.groupby(["StockCode","Description"]).size().reset_index(name="n").sort_values(["StockCode","n"], ascending=[True,False]).drop_duplicates("StockCode").set_index("StockCode")[["Description"]]
    task_progress.update(1)
df_product = desc.join(tmp).reset_index()
df_product["std_of_price"] = df_product["std_of_price"].fillna(0)
df_product.head()

#df.drop(columns='Description', inplace=True)

# Saving product catalog
if not os.path.exists('data/given_data/prod_catalog_holdout.csv'):
    with tqdm(total=1, desc='Saving prod_catalog_holdout.csv', unit="task", leave=False) as task_progress:
        df_product.to_csv('data/given_data/prod_catalog_holdout.csv', index=False)
        task_progress.update(1)
if not os.path.exists('data/given_data/prod_catalog_holdout.parquet'):
    with tqdm(total=1, desc='Saving prod_catalog_holdout.parquet', unit="task", leave=False) as task_progress:
        df_product.to_parquet('data/given_data/prod_catalog_holdout.parquet', index=False)
        task_progress.update(1)

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
#from tqdm import tqdm
#import re
import os
#from tqdm.notebook import tqdm
from tqdm.auto import tqdm
from collections import defaultdict

#from sentence_transformers import SentenceTransformer
#from sklearn.cluster import KMeans, MiniBatchKMeans, AgglomerativeClustering
#from sklearn.metrics import silhouette_score

print('Current working directory is', os.getcwd())

# Loading calibration data
with tqdm(total=1, desc='Reading calibration_raw.csv', unit="task", leave=False) as task_progress:
    df = pd.read_csv("data/given_data/calibration_raw.csv")
    task_progress.update(1)

df.head(3)

df['Invoice'] = df['Invoice'].astype(str).str.upper().str.strip()
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
df = df.rename(columns={"Customer ID": "Customer_ID"})
df["Customer_ID"] = df["Customer_ID"].astype("Int64")
df['StockCode'] = df['StockCode'].astype(str).str.upper().str.strip()
df['Description'] = df['Description'].astype(str).str.upper().str.strip()
df['Country'] = df['Country'].astype(str).str.upper().str.strip()

df.info()

df.info()

df.info()

# Both negative price and negative quantity mean cancellation of orders; 
# I standardize cancellation of order by considering Quantity -ve and price positive
df['Quantity'] = df['Quantity'] * np.sign(df['Price'])
df['Price'] = np.abs(df['Price'])

# Sort by time
df = df.sort_values("InvoiceDate").reset_index(drop=True)

# Status column
df["status"] = pd.NA

# ---------------------------------------------------
# Matching structure:
# key = (Customer_ID, StockCode, abs(Quantity), Price)
# value = list of unmatched positive row positions
# ---------------------------------------------------
open_sales = defaultdict(list)

# Use tqdm for progress tracking
for i, row in tqdm(
    enumerate(df.itertuples(index=False)),
    total=len(df),
    desc="Matching returns"
):
    key = (
        row.Customer_ID,
        row.StockCode,
        abs(row.Quantity),
        row.Price
    )
    if row.Quantity > 0:
        open_sales[key].append(i)

    elif row.Quantity < 0:
        if open_sales[key]:
            match_i = open_sales[key].pop()
            df.at[match_i, "status"] = "cancelled"
            df.at[i, "status"] = "return: purchase found"
        else:
            df.at[i, "status"] = "return: purchase not found"

# Any remaining positive rows are completed purchases
df.loc[(df["Quantity"] > 0) & (df["status"].isna()), "status"] = "completed"

df['status'] = df['status'].astype(str)
df.info()

## Keeping only completed orders
df = df[df['status']=='completed']

print(f'Size of the calibration data after adjusting returned order is: {len(df)}')

# droping status column
df.drop(columns = 'status', inplace=True)

# Deleting lines that put as a test, or as fee	
df = df[~df["Description"].isin(["CHECKED", "CHECK", "CHECK?", "STOCK CHECK", "CARRIAGE"])]
df = df[~df["Description"].str.contains("test|fee|damaged|BANK CHARGES|DISCOUNT|POSTAGE|MANUAL|ADJUST|test|lost|samples", case=False, na=False)]

# 5. Remove administrative/service stock codes
non_product_codes = ['DOT', 'POST', 'C2', 'M', 'BANK CHARGES', 'PADS', 'CRUK']
df = df[~df['StockCode'].isin(non_product_codes)]

###  Checking if customer and country mismatch exists

# Creating new dataframe by dropping rows with blank customer_ID for checking customer country mismatch check
rows_with_customer = df[df['Customer_ID'].notna()].copy()

# Count unique Country per Customer_ID
with tqdm(total=1, desc='Checking customer countries', unit="task", leave=False) as task_progress:
    customer_country_check = (
        rows_with_customer
        .groupby('Customer_ID')
        .agg(
            n_countries=('Country', lambda x: x.nunique(dropna=False))
        )
        .reset_index()
    )
    task_progress.update(1)

# Customers linked to more than one country
country_mismatch = customer_country_check[
    customer_country_check['n_countries'] > 1
]

# Save all original rows belonging to those multiple country customers
customer_country_mismatches = (
    rows_with_customer[
        rows_with_customer['Customer_ID'].isin(country_mismatch['Customer_ID'])
    ]
    .sort_values(['Customer_ID', 'Country', 'InvoiceDate'])
)

if country_mismatch.shape[0] == 0:
    print('No mismatch between Customer_ID and Country')
else:
    print(f'❌ Country column of {country_mismatch.shape[0]} customers are not unique between different lines. \n \n ID\'s of the customres are:\n')
    print(customer_country_mismatches[
    ['Customer_ID', 'Country']
].drop_duplicates().sort_values(['Customer_ID', 'Country']).head(50))

### Resolving mismatch between country and customer

# Invoice count and total quantity by Customer_ID
with tqdm(total=1, desc='Aggregating customer countries', unit="task", leave=False) as task_progress:
    customer_country_stats = (
        rows_with_customer
        .groupby(['Customer_ID', 'Country'])
        .agg(
            n_invoices=('Invoice', 'nunique'),
            total_quantity=('Quantity', 'sum')
        )
        .reset_index()
    )
    task_progress.update(1)

# Pick mode country:
# 1. highest number of invoices
# 2. if tied, highest total quantity
mode_country_per_customer = (
    customer_country_stats
    .sort_values(
        ['Customer_ID', 'n_invoices', 'total_quantity'],
        ascending=[True, False, False]
    )
    .drop_duplicates('Customer_ID', keep='first')
    [['Customer_ID', 'Country']]
    .rename(columns={'Country': 'mode_country'})
)

# Add mode country to original df
df = df.merge(
    mode_country_per_customer,
    on='Customer_ID',
    how='left'
)

# Replace Country only for rows with given Customer_ID
df.loc[df['Customer_ID'].notna(), 'Country'] = df.loc[
    df['Customer_ID'].notna(),
    'mode_country'
]

# Drop helper column
df = df.drop(columns='mode_country')

print('Mismatches resolved between Customer_ID and Country, by updating country column. ✅')

### Checking whether rows corresponds to a same invoice referes to a same Customer_ID 

# Count unique Customer_ID  per Invoice
# dropna=False means missing Customer_ID is counted as its own value
with tqdm(total=1, desc='Checking invoice customers', unit="task", leave=False) as task_progress:
    invoice_check = (
        df.groupby('Invoice')
          .agg(
              n_customers=('Customer_ID', lambda x: x.nunique(dropna=False))
          )
          .reset_index()
    )
    task_progress.update(1)

# Invoices where same invoice has different customer 
bad_invoices = invoice_check[(invoice_check['n_customers'] > 1)]

# Save all original rows belonging to those problematic invoices
invoice_exceptions = (
    df[df['Invoice'].isin(bad_invoices['Invoice'])]
    .sort_values(['Invoice', 'Customer_ID'])
)


if bad_invoices.shape[0] == 0:
    print('No mismatch found between invoice and Customer_ID.  ✅')
else:
    print(f'{bad_invoices.shape[0]} mismatches observed between invoice and Customer_ID. Some mismatches as examples are:')
    print(invoice_exceptions[['Invoice', 'Customer_ID', 'Country']].drop_duplicates().head(5))

# Number of duplicate rows (considering all columns)
n_duplicates = df.duplicated().sum()

if n_duplicates == 0:
    print("✅ No duplicate rows found.")
else:
    print(f"❌ Found {n_duplicates:,} duplicate rows.")

    # Drop duplicates
    df = df.drop_duplicates().reset_index(drop=True)

    print(f"Deleted {n_duplicates:,} duplicate rows.")
    print(f"Remaining rows: {len(df):,}")

# Total rows
n_rows = len(df)

# Number of unique (Invoice, StockCode) pairs
n_unique = df[['Invoice', 'StockCode']].drop_duplicates().shape[0]

print(f"Total rows: {n_rows:,}")
print(f"Unique (Invoice, StockCode) pairs: {n_unique:,}")

if n_rows == n_unique:
    print("✅ Now, every (Invoice, StockCode)  is unique.")
else:
    print(f"❌ Found {n_rows - n_unique:,} duplicate (Invoice, StockCode) pairs.")

df["line_amount"] = df["Quantity"] * df["Price"]

with tqdm(total=1, desc='Grouping transaction lines', unit="task", leave=False) as task_progress:
    df = (
        df.groupby(["Invoice", "StockCode"], as_index=False)
          .agg(
              Quantity=("Quantity", "sum"),
              line_amount=("line_amount", "sum"),
              Description=("Description", "first"),
              InvoiceDate=("InvoiceDate", "first"),
              **{"Customer_ID": ("Customer_ID", "first")},
              Country=("Country", "first")
          )
    )
    task_progress.update(1)

df["Price"] = df["line_amount"] / df["Quantity"]

# Total rows
n_rows = len(df)

# Number of unique (Invoice, StockCode) pairs
n_unique = df[['Invoice', 'StockCode']].drop_duplicates().shape[0]

if n_rows == n_unique:
    print("After cleaning, (Invoice, StockCode)  is unique for every lines. ✅")
    print(f"Total lines after handling duplicates is: {n_unique:,}")
else:
    print(f"❌ Found {n_rows - n_unique:,} duplicate (Invoice, StockCode) pairs from total rows: {n_rows:,}.")

with tqdm(total=1, desc='Standardizing product descriptions', unit="task", leave=False) as task_progress:
    collapsed = (
        df.groupby(['StockCode', 'Description'])
          .size()
          .rename('count')
          .reset_index()
    )
    task_progress.update(1)

with tqdm(total=1, desc='Grouping data', unit="task", leave=False) as task_progress:
    collapsed['rank'] = (
        collapsed.groupby('StockCode')['count']
                 .rank(method='first', ascending=False)
    )
    task_progress.update(1)

collapsed = collapsed[collapsed['rank'] == 1]

df = df.merge(collapsed, on='StockCode', how='left', suffixes=('', '_top'))
df['Description'] = df['Description_top']
df = df.drop(columns=['Description_top', 'rank', 'count'])

#df.drop(columns='Description', inplace=True)
if not os.path.exists('data/working_data/calib_all_transaction.csv'):
    with tqdm(total=1, desc='Saving calib_all_transaction.csv', unit="task", leave=False) as task_progress:
        df.to_csv('data/working_data/calib_all_transaction.csv', index=False)
        task_progress.update(1)

# Number of rows before removing rows with missing Customer_ID
print(f"Rows before removing rows with missing Customer_ID : {len(df)}")

# Remove rows with missing Customer_ID
df = df[df["Customer_ID"].notna()].copy()

print(f"Rows remaining                                     : {len(df)}")

df.info()

## creating invoice table: Group by customer and invoice, then sum the costs and quantities
with tqdm(total=1, desc='Aggregating calibration invoices', unit="task", leave=False) as task_progress:
    df_invoice = df.groupby(['Invoice', 'Customer_ID','InvoiceDate', 'Country']).agg(
        total_amount=('line_amount', 'sum'),
        number_of_items=('Quantity', 'sum')
    ).reset_index()
    task_progress.update(1)

# make a copy
df_order = df_invoice.copy()

# extract date only from date-time
df_order["order_date"] = df_order["InvoiceDate"].dt.date

# group transation lines with same customer + same day into one order
with tqdm(total=1, desc='Aggregating calibration orders', unit="task", leave=False) as task_progress:
    df_order = (
        df_order
        .groupby(["Customer_ID", "order_date"], as_index=False)
        .agg(
            total_amount=("total_amount", "sum"),
            number_of_items=("number_of_items", "sum"),
            Country=("Country", "first"),
            InvoiceDate=('InvoiceDate', 'first')
        )
    )
    task_progress.update(1)

# create order_id as string: Customer_ID + date
df_order["order_id"] = (
    df_order["Customer_ID"].astype(str) + "_" +
    pd.to_datetime(df_order["order_date"]).dt.strftime("%Y-%m-%d")
)

df_order.drop(columns='InvoiceDate', inplace=True)

df_order['order_date'] = pd.to_datetime(df_order['order_date'])

print('df_order')
df_order.info()

if not os.path.exists('data/working_data/calib_order.csv'):
    with tqdm(total=1, desc='Saving calib_order.csv', unit="task", leave=False) as task_progress:
        df_order.to_csv('data/working_data/calib_order.csv', index=False)
        task_progress.update(1)

with tqdm(total=1, desc='Reading holdout_transaction.csv', unit="task", leave=False) as task_progress:
    df_holdout = pd.read_csv('data/given_data/holdout_transaction.csv')
    task_progress.update(1)
df_holdout['InvoiceDate'] = pd.to_datetime(df_holdout['InvoiceDate'])

df_holdout.info()

# Number of rows before removing rows with missing Customer_ID
print(f"Rows before removing rows with missing Customer_ID : {len(df_holdout)}")

# Remove rows with missing Customer_ID
df_holdout = df_holdout[df_holdout["Customer_ID"].notna()].copy()

print(f"Rows remaining                                     : {len(df_holdout)}")

df_holdout.info()

# Line amount
df_holdout["line_amount"] = df_holdout["Quantity"] * df_holdout["Price"]

## creating invoice table: Group by customer and invoice, then sum the costs and quantities
with tqdm(total=1, desc='Aggregating holdout invoices', unit="task", leave=False) as task_progress:
    df_holdout_invoice = df_holdout.groupby(['Invoice', 'Customer_ID','InvoiceDate', 'Country']).agg(
        total_amount=('line_amount', 'sum'),
        number_of_items=('Quantity', 'sum')
    ).reset_index()
    task_progress.update(1)

# make a copy
df_holdout_order = df_holdout_invoice.copy()

# extract date only from date-time
df_holdout_order["order_date"] = df_holdout_order["InvoiceDate"].dt.date

# group transation lines with same customer + same day into one order
with tqdm(total=1, desc='Aggregating holdout orders', unit="task", leave=False) as task_progress:
    df_holdout_order = (
        df_holdout_order
        .groupby(["Customer_ID", "order_date"], as_index=False)
        .agg(
            total_amount=("total_amount", "sum"),
            number_of_items=("number_of_items", "sum"),
            Country=("Country", "first"),
            InvoiceDate=('InvoiceDate', 'first')
        )
    )
    task_progress.update(1)

# create order_id as string: Customer_ID + date
df_holdout_order["order_id"] = (
    df_holdout_order["Customer_ID"].astype(str) + "_" +
    pd.to_datetime(df_holdout_order["order_date"]).dt.strftime("%Y-%m-%d")
)

df_holdout_order.drop(columns='InvoiceDate', inplace=True)

df_holdout_order['order_date'] = pd.to_datetime(df_holdout_order['order_date'])

print('df_holdout_order')
df_holdout_order.info()

if not os.path.exists('data/working_data/holdout_order.csv'):
    with tqdm(total=1, desc='Saving holdout_order.csv', unit="task", leave=False) as task_progress:
        df_holdout_order.to_csv('data/working_data/holdout_order.csv', index=False)
        task_progress.update(1)

# Product recommendations from 04_Product_reco.ipynb

import pandas as pd

import numpy as np

import re

import os

from tqdm.auto import tqdm

from collections import defaultdict

with tqdm(total=1, desc='Importing sentence transformers', unit="task", leave=False) as task_progress:
    from sentence_transformers import SentenceTransformer
    task_progress.update(1)

with tqdm(total=1, desc='Importing PyTorch', unit="task", leave=False) as task_progress:
    import torch
    task_progress.update(1)

import torch.nn as nn

import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader

with tqdm(total=1, desc='Reading calib_all_transaction.csv', unit="task", leave=False) as task_progress:
    df_calib = pd.read_csv('data/working_data/calib_all_transaction.csv')
    task_progress.update(1)
with tqdm(total=1, desc='Reading holdout_transaction.csv', unit="task", leave=False) as task_progress:
    df_holdout = pd.read_csv('data/given_data/holdout_transaction.csv')
    task_progress.update(1)
with tqdm(total=1, desc='Reading prod_catalog_calib.csv', unit="task", leave=False) as task_progress:
    df_catalog_calib = pd.read_csv('data/given_data/prod_catalog_calib.csv')
    task_progress.update(1)
with tqdm(total=1, desc='Reading prod_catalog_holdout.csv', unit="task", leave=False) as task_progress:
    df_catalog_holdout = pd.read_csv('data/given_data/prod_catalog_holdout.csv')
    task_progress.update(1)

df_calib['Invoice'] = df_calib['Invoice'].astype(str).str.upper().str.strip()
df_calib['InvoiceDate'] = pd.to_datetime(df_calib['InvoiceDate'])
df_calib["Customer_ID"] = df_calib["Customer_ID"].astype("Int64")

df_holdout['Invoice'] = df_holdout['Invoice'].astype(str).str.upper().str.strip()
df_holdout['InvoiceDate'] = pd.to_datetime(df_holdout['InvoiceDate'])
df_holdout["Customer_ID"] = df_holdout["Customer_ID"].astype("Int64")

print('df_calib')
df_calib.info()

print('df_holdout')
df_holdout.info()

print('df_catalog_calib')
df_catalog_calib.info()

print('df_catalog_holdout')
df_catalog_holdout.info()

# ==========================================================
# Segnmenting customers based on how often a customer buy a new product
# To measure such behavior, I calculate a new metric named 'product_transition_rate' from calibration/ train data
# For each customer:
#   - sort invoices chronologically
#   - first invoice is used only to initialize history and product transition rate calculated from invoice 2 to n    
# Outputs: df_transition_rate
# ==========================================================

def customer_segmentation(df_history):

    # Build invoice-level baskets: one row per customer-invoice-date
    with tqdm(total=1, desc='Building customer invoice histories', unit="task", leave=False) as task_progress:
        df_invoice_level = (
            df_history.groupby(["Customer_ID", "Invoice", "InvoiceDate"], as_index=False)["StockCode"]
              .agg(lambda s: set(s))
              .rename(columns={"StockCode": "product_set"})
        )
        task_progress.update(1)
    
    # Sort chronologically within each customer
    df_invoice_level = df_invoice_level.sort_values(
        ["Customer_ID", "InvoiceDate", "Invoice"]
    ).reset_index(drop=True)
    
    invoice_results = []
    transition_rate = []
    
    for customer_id, invoice in tqdm(df_invoice_level.groupby("Customer_ID", sort=False), desc='Segmenting customers', unit="customer", leave=False):
        product_set_cumul = set()
        individual_metric = []
    
        invoice = (
            invoice
            .sort_values(["InvoiceDate", "Invoice"])
            .reset_index(drop=True)
        )
        
        for invoice_pos, row in invoice.iterrows():
            invoice_size = len(row["product_set"])
    
            # Skip scoring the first invoice because there is no history history
            if len(product_set_cumul) > 0 and invoice_size > 0:
                new_products = row["product_set"] - product_set_cumul
                
                # invoice_pos = 1 for second invoice, 2 for third, ...
                weight = 3.3 - 2.2 * np.exp(-0.1 * invoice_pos)
                
                individual_metric.append(weight * len(new_products) / invoice_size)
    
            # Update product_set_cumul after scoring; update operation on the set work like an union
            product_set_cumul.update(row["product_set"])
    
        product_transition_rate = np.mean(individual_metric) if len(individual_metric) > 0 else np.nan
    
        transition_rate.append({
            "Customer_ID": customer_id,
            "number_of_invoice": invoice_pos+1,
            "product_transition_rate": product_transition_rate
        })
    
    # Final outputs
    df_transition_rate = pd.DataFrame(transition_rate).sort_values(
        "product_transition_rate", ascending=False
    ).reset_index(drop=True)
    
    # ==========================================================
    # Customer segmentation by product_transition_rate
    # similar product shoppers (dynamic shopper)   : product_transition_rate < 0.65 (>=0.65); 
    # infrequent shopper: product_transition_rate is null / unknown
    # ==========================================================
    
    threshold = 0.65
    
    df_transition_rate["shopper_type"] = np.select(
        [
            df_transition_rate["product_transition_rate"].isna(),
            df_transition_rate["product_transition_rate"] < threshold,
            df_transition_rate["product_transition_rate"] >= threshold,
        ],
        [
            "infrequent_shopper",
            "similar_prod_shoppers",
            "product_explorer",
        ],
        default="infrequent_shopper"
    )
    
    similar_prod_shoppers = set(
        df_transition_rate.loc[
            df_transition_rate["shopper_type"] == "similar_prod_shoppers",
            "Customer_ID"
        ]
    )
    
    product_explorers = set(
        df_transition_rate.loc[
            df_transition_rate["shopper_type"] == "product_explorer",
            "Customer_ID"
        ]
    )
    
    infrequent_shoppers = set(
        df_transition_rate.loc[
            df_transition_rate["shopper_type"] == "infrequent_shopper",
            "Customer_ID"
        ]
    )

    return similar_prod_shoppers, product_explorers, infrequent_shoppers, df_transition_rate

_,_,_,df_transition_rate = customer_segmentation(df_calib )

if not os.path.exists('data/working_data/customer_category.csv'):
    df_customer_category = df_transition_rate[['Customer_ID', 'shopper_type']].copy()
    df_customer_category['shopper_type'] = df_customer_category['shopper_type'].map({
        'infrequent_shopper': 0,
        'similar_prod_shoppers': 1,
        'product_explorer': 2,
    })
    df_customer_category = df_customer_category.rename(columns={'shopper_type': 'customer_category'})
    with tqdm(total=1, desc='Saving customer_category.csv', unit="task", leave=False) as task_progress:
        df_customer_category.to_csv('data/working_data/customer_category.csv', index=False)
        task_progress.update(1)

def short_descript_v2(text):
    text = str(text).lower()

    # Remove measurements like 30cm, 500ml, 2kg
    text = re.sub(r'\b\d+\s*(cm|mm|m|inch|inches|oz|ml|l|kg|g)\b', ' ', text)

    # Remove remaining standalone numbers
    text = re.sub(r'\b\d+\b', ' ', text)

    # Remove punctuation
    text = re.sub(r'[^a-z\s]', ' ', text)

    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()

    # Remove only words that rarely define the product
    noise_words = {
        'set', 'pack', 'piece', 'pieces',
        'small', 'large', 'medium', 'mini', 'big',
        'assorted',
        'colour', 'colours',
        'red', 'blue', 'green', 'pink', 'white',
        'black', 'yellow', 'purple', 'orange', 'brown'
    }

    words = text.split()
    words = [w for w in words if w not in noise_words]

    return " ".join(words)

df_catalog_all = (
    pd.concat([
        df_catalog_calib[["StockCode", "Description"]],
        df_catalog_holdout[["StockCode", "Description"]]
    ], ignore_index=True)
    .drop_duplicates(subset="StockCode", keep="first")
    .reset_index(drop=True)
)


df_catalog_all['short_description2'] = df_catalog_all['Description'].apply(short_descript_v2)

with tqdm(total=1, desc='Loading MiniLM embedding model', unit="task", leave=False) as task_progress:
    model = SentenceTransformer("all-MiniLM-L6-v2")
    task_progress.update(1)

embeddings_short2 = model.encode(
    df_catalog_all['short_description2'].tolist(),
    show_progress_bar=True
)

df_catalog_all["embeddings_short2"] = list(embeddings_short2)

del model

def repeat_model(df_history, df_catalog_history, df_catalog_future):

    customer_product = (
        df_history
        .dropna(subset=["Customer_ID"])
        .groupby(["Customer_ID", "StockCode"])
        .agg(
            frequency=("Invoice", "count"),
            quantity=("Quantity", "sum"),
            spend=("line_amount", "sum"),
            last_purchase=("InvoiceDate", "max")
        )
        .reset_index()
    )

    '''
    customer_product = customer_product[
        customer_product["StockCode"].isin(df_catalog_future["StockCode"])
    ].copy()
    '''
    
    customer_product = customer_product.sort_values(
        ["Customer_ID", "frequency", "last_purchase"],
        ascending=[True, False, False]
    )
    
    ranked_recommendation_dict = (
        customer_product
        .groupby("Customer_ID")["StockCode"]
        .apply(list)
        .to_dict()
    )

    product_score_dict = {
        customer: {
            stockcode: rank
            for rank, stockcode in enumerate(
                group["StockCode"],
                start=1
            )
        }
        for customer, group in customer_product.groupby("Customer_ID", sort=False)
    }
    
    return ranked_recommendation_dict, product_score_dict

# ==========================================================
# creating class for passing into DataLoader(); DataLoader() creates batch
# ==========================================================

# dataset_pair inherits from 'Dataset' class
class dataset_pair(Dataset):

    def __init__(
        self,
        df_future, # used to create all positive interaction; 
        df_catalog_future, # -ve interaction \in [set (df_catalog_future)- purchase_set_per_cust)
        customer_id_to_index,
        stockcode_to_index,
        num_neg=4
    ):

        df_future = df_future.dropna(subset=["Customer_ID"]).copy()
        
        # collapsing df_future
        with tqdm(total=1, desc='Aggregating two-tower training purchases', unit="task", leave=False) as task_progress:
            cust_prod_future = (
                df_future
                .groupby(["Customer_ID", "StockCode"])
                .agg(
                    frequency=("Invoice", "count"),
                    quantity=("Quantity", "sum"),
                    # spend=("line_amount", "sum"),
                    last_purchase=("InvoiceDate", "max")
                )
                .reset_index()
            )
            task_progress.update(1)

        # purchase_set is going to be used to create -ve pairs 
        purchase_set_per_cust = {}

        # loop for building purchase_set_per_cust 
        for customer_id, prod_info in tqdm(cust_prod_future.groupby("Customer_ID"), desc='Preparing negative-sampling histories', unit="customer", leave=False):
            cust_index = customer_id_to_index[customer_id]
            product_index_list = prod_info["StockCode"].map(stockcode_to_index).to_list()
            purchase_set_per_cust[cust_index] = set(product_index_list)

        future_catlg_size = len(df_catalog_future['StockCode'])

        # pair_list stores both positive and negative pair of customer_index vs product_index
        self.pair_list = []
        self.num_neg = num_neg

        # Training pairs (positive interactions)
        df_pos_pairs = cust_prod_future[["Customer_ID", "StockCode"]].drop_duplicates().copy()
        df_pos_pairs["cust_index"] = df_pos_pairs["Customer_ID"].map(customer_id_to_index)
        df_pos_pairs["prod_index"] = df_pos_pairs["StockCode"].map(stockcode_to_index)

        for row in tqdm(
            df_pos_pairs.itertuples(index=False),
            total=len(df_pos_pairs),
            desc="Building train samples",
            leave=False
        ):
            cust_index = row.cust_index
            prod_index = row.prod_index
            
            # appending positive sample ((cust_index, prod_index, if_postv))
            self.pair_list.append((cust_index, prod_index, 1.0))

            # some (num_neg=4) negative samples are picked randomly against each positive sample
            purchase_set = purchase_set_per_cust[cust_index]
            
            for _ in range(num_neg):
                random = np.random.randint(0, future_catlg_size)
                neg_prod_index = stockcode_to_index[
                    df_catalog_future.iloc[random]["StockCode"]
                ]
            
                while neg_prod_index in purchase_set:
                    random = np.random.randint(0, future_catlg_size)
                    neg_prod_index = stockcode_to_index[
                        df_catalog_future.iloc[random]["StockCode"]
                    ]
            
                self.pair_list.append((cust_index, neg_prod_index, 0.0))

    def __len__(self):
        return len(self.pair_list)

    def __getitem__(self, idx):
        return self.pair_list[idx]

# ==========================================================
#  Two-Tower model:   Version 1:
#    - customer tower embedding training will be performed
#    - product tower embedding remain constant
# ==========================================================
class Model_2Tower(nn.Module):
    
    def __init__(
        self,
        product_embed_matrix,
        product_embed_dim,
        dim_layer_2=128,
        dim_layer_3=128
    ):

        super().__init__()

        # Product embedding should not be trained, instead saved as register_buffer
        self.register_buffer("product_embed_tensor", torch.tensor(product_embed_matrix, dtype=torch.float32))

        # purchase history tower
        self.history_tower = nn.Sequential(
            nn.Linear(product_embed_dim, dim_layer_2),
            nn.ReLU(),
            nn.Linear(dim_layer_2, dim_layer_3),
            #nn.Sigmoid(),
        )

        # product tower
        self.product_tower = nn.Sequential(
            nn.Linear(product_embed_dim, dim_layer_2),
            nn.ReLU(),
            nn.Linear(dim_layer_2, dim_layer_3),
            #nn.Sigmoid(),
        )

        # logit scale =====================
        self.logit_scale = nn.Parameter(torch.tensor(10.0))

    def encode_history_tower(self, hist_products, hist_weights):

        product_hist_emb = self.product_embed_tensor[hist_products]

        w = hist_weights.unsqueeze(-1)

        product_hist = (
            product_hist_emb * w
        ).sum(dim=1) / w.sum(dim=1).clamp(min=1e-6)

        tower1_output = self.history_tower(product_hist)
        return F.normalize(tower1_output, dim=-1)

    def encode_product_tower(self, prod_index_tensor):
        prod_tensor = self.product_embed_tensor[prod_index_tensor]   
        tower2_output = self.product_tower(prod_tensor)                 
        return F.normalize(tower2_output, dim=-1)

    def forward(self, prod_index_tensor, hist_products, hist_weights):
        t1 = self.encode_history_tower(hist_products, hist_weights)
        t2 = self.encode_product_tower(prod_index_tensor)
        logits = self.logit_scale * (t1 * t2).sum(dim=-1)
        return logits

# ==========================================================
# Defining Training part
# ==========================================================

def train_2Tower(
    df_history, df_future, df_catalog_history, df_catalog_future, 
    df_catalog_all, EPOCHS = 8, dim_layer_2=128, dim_layer_3=64
):

    # ==========================================================
    # Preparing data
    # ==========================================================

    df_history = df_history.dropna(subset=["Customer_ID"]).copy()
    df_future = df_future.dropna(subset=["Customer_ID"]).copy()

    # creation of normalized product embedding
    embedding_col = "embeddings_short2"
    
    product_embed_matrix = np.vstack(df_catalog_all[embedding_col])
    product_embed_dim = product_embed_matrix.shape[1]
    normlzsn_constant = np.linalg.norm(product_embed_matrix, axis=1, keepdims=True)
    normlzsn_constant[normlzsn_constant == 0] = 1.0
    product_embed_matrix = product_embed_matrix / normlzsn_constant
    
    # adding one more row that corresponds to padding
    product_embed_matrix = np.vstack([product_embed_matrix, np.zeros((1, product_embed_dim), dtype=np.float32)]) 
    
    # stockcode_list
    stockcode_list = df_catalog_all["StockCode"].tolist()

    # pointer to padding row in product_embed_matrix
    pad_embed_index = len(stockcode_list)  
    
    # mapping product stockcode <-> index; index is used to access embedding
    stockcode_list = df_catalog_all["StockCode"].tolist()
    stockcode_to_index = {s: i for i, s in enumerate(stockcode_list)}
    index_to_stockcode = {i: s for s, i in stockcode_to_index.items()}

    # common customer both in history an future
    customer_id_list = sorted(
        set(df_history["Customer_ID"]) &
        set(df_future["Customer_ID"])
    )

    # mapping customer <-> index; index is used to access history data    
    customer_id_to_index = {c: i for i, c in enumerate(customer_id_list)}

    # keeping only common customer
    df_history = df_history[
        df_history["Customer_ID"].isin(customer_id_list)
    ].copy()

    df_future = df_future[
        df_future["Customer_ID"].isin(customer_id_list)
    ].copy()

    # collapsing df_history for creating purchase_dict_per_cust[
    with tqdm(total=1, desc='Aggregating two-tower purchase histories', unit="task", leave=False) as task_progress:
        df_cust_prod_history = (
            df_history
            .groupby(["Customer_ID", "StockCode"])
            .agg(
                frequency=("Invoice", "count"),
                quantity=("Quantity", "sum"),
                # spend=("line_amount", "sum"),
                last_purchase=("InvoiceDate", "max")
            )
            .reset_index()
        )
        task_progress.update(1)
    
    weight = "frequency"

    # purchase_dict is a nested dict with the inner dict = (poduct_index:weight)
    # purcahse dict is going to be used to add purchase history (products with weights) to batch_for_train1ing
    purchase_dict_per_cust = {}
    
    for customer_id, prod_info in tqdm(df_cust_prod_history.groupby("Customer_ID"), desc='Collecting two-tower customer histories', unit="customer", leave=False):
        cust_index = customer_id_to_index[customer_id]
        product_index_list = prod_info["StockCode"].map(stockcode_to_index).to_list()
        weight_list = prod_info[weight].to_list()    
        purchase_dict_per_cust[cust_index] = {i: w for i, w in zip(product_index_list, weight_list)}

    
    # creating training batch (history part)
    dataset = dataset_pair(
        df_future,
        df_catalog_future,
        customer_id_to_index,
        stockcode_to_index
    )

    batch_for_train1ing = DataLoader(
        dataset,
        batch_size=1024,
        shuffle=True,
        num_workers=0
    )

    # ==========================================================
    # initializing model
    # ==========================================================

    model = Model_2Tower(
        product_embed_matrix,
        product_embed_dim,
        dim_layer_2=dim_layer_2,
        dim_layer_3=dim_layer_3
    )
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    criterion = nn.BCEWithLogitsLoss()
    # ==========================================================
    # Training
    # ==========================================================
    EPOCHS = EPOCHS

    for epoch in range(EPOCHS):

        model.train()
        total_loss = 0.0
        for cust_index_tensor, prod_index_tensor, if_pos_tensor in tqdm(
            batch_for_train1ing,
            desc=f"Epoch {epoch+1}/{EPOCHS}",
            leave=False
        ):
        
            hist_products_list = []
            hist_weights_list = []
        
            # Add history to the current batch
            for cust_index in cust_index_tensor:

                cust_index = int(cust_index)
                
                purchase_history = purchase_dict_per_cust[cust_index]

                # building lists for purcahse history and weights for each customer
                purchase_list = list(purchase_history.keys())
                weights = [
                    purchase_history[product_index]
                    for product_index in purchase_list
                ]

                # appending individual history list to the total list
                hist_products_list.append(purchase_list)
                hist_weights_list.append(weights)
        
            # determining maximum length of purchase history
            max_batch_history = max(
                len(purchase_list)
                for purchase_list in hist_products_list
            )

            # Padding histories within the current batch
            for i in range(len(hist_products_list)):
                padding_length = max_batch_history - len(hist_products_list[i])
        
                hist_products_list[i] += [pad_embed_index] * padding_length
                hist_weights_list[i] += [0.0] * padding_length
        
            # Convert the batch histories to tensors
            hist_products = torch.tensor(
                hist_products_list,
                dtype=torch.long
            )
        
            hist_weights = torch.tensor(
                hist_weights_list,
                dtype=torch.float32
            )
        
            optimizer.zero_grad()

            # calling forward function        
            logits = model(
                prod_index_tensor,
                hist_products,
                hist_weights
            )

            if_pos_tensor = if_pos_tensor.float()
            loss = criterion(logits, if_pos_tensor)
        
            loss.backward()
            optimizer.step()
        
            total_loss += loss.item() * if_pos_tensor.size(0)

        avg_loss = total_loss / len(batch_for_train1ing.dataset)
        print(f"Training: epoch {epoch+1}/{EPOCHS} | loss = {avg_loss:.4f}")

    return (
        model,
        stockcode_to_index,
        index_to_stockcode,
        pad_embed_index
    )

# ==========================================================
# Defining recommendation func
# ==========================================================

def recommend_2Tower(
    df_history,
    df_catalog_future,
    df_catalog_all,
    trained_model,
    stockcode_to_index,
    index_to_stockcode,
    pad_embed_index,
    customer_id_to_matrix_index,
    product_code_to_matrix_index
):

    df_history = df_history.dropna(subset=["Customer_ID"]).copy()

    # common customer both in history an future
    customer_id_list = sorted(
        set(df_history["Customer_ID"])
    )
    
    # mapping customer <-> index; index is used to access history data   
    customer_id_to_index = {c: i for i, c in enumerate(customer_id_list)}
    customer_indices = list(customer_id_to_index.values())

    trained_model.eval()
    
    # Disable gradient tracking by no_grad() for evaluation stage
    with torch.no_grad():
    
        # Create integer indices for all products: 0, 1, 2, ..., number_of_products-1
        all_prod_index_tensor = torch.arange(
            len(df_catalog_all),
            dtype=torch.long
        )
    
        # Pass all products through the trained product tower; Output: one latent vector for each product
        with tqdm(total=1, desc='Encoding two-tower catalogue', unit="task", leave=False) as task_progress:
            product_latent = trained_model.encode_product_tower(all_prod_index_tensor).numpy()
            task_progress.update(1)

    # collapsing df_history for building purchase history
    with tqdm(total=1, desc='Aggregating two-tower purchase histories', unit="task", leave=False) as task_progress:
        df_cust_prod_history = (
            df_history
            .groupby(["Customer_ID", "StockCode"])
            .agg(
                frequency=("Invoice", "count"),
                quantity=("Quantity", "sum"),
                # spend=("line_amount", "sum"),
                last_purchase=("InvoiceDate", "max")
            )
            .reset_index()
        )
        task_progress.update(1)
    
    weight = "frequency"

    # purchase_dict is a nested dict with the inner dict = (poduct_index:weight)
    # purcahse dict is going to be used to add purchase history (products with weights) to batch_for_train1ing
    purchase_dict_per_cust = {}
    
    for customer_id, prod_info in tqdm(df_cust_prod_history.groupby("Customer_ID"), desc='Collecting two-tower customer histories', unit="customer", leave=False):
        cust_index = customer_id_to_index[customer_id]
        product_index_list = prod_info["StockCode"].map(stockcode_to_index).to_list()
        weight_list = prod_info[weight].to_list()    
        purchase_dict_per_cust[cust_index] = {i: w for i, w in zip(product_index_list, weight_list)}

    hist_products_list = []
    hist_weights_list = []
    
    # Build history
    for cust_index in tqdm(customer_indices, desc='Preparing two-tower history lists', unit="customer", leave=False):

        cust_index = int(cust_index)

        purchase_history = purchase_dict_per_cust[cust_index]

        purchase_list = list(purchase_history.keys())

        weights = [
            purchase_history[product_index]
            for product_index in purchase_list
        ]

        hist_products_list.append(purchase_list)
        hist_weights_list.append(weights)

    # Pad histories
    max_batch_history = max(
        len(purchase_list)
        for purchase_list in hist_products_list
    )
    for i in tqdm(range(len(hist_products_list)), desc='Padding two-tower histories', unit="customer", leave=False):
        padding_length = max_batch_history - len(hist_products_list[i])

        hist_products_list[i] += [pad_embed_index] * padding_length
        hist_weights_list[i] += [0.0] * padding_length

    # Convert the batch history lists to tensors
    with tqdm(total=1, desc='Creating two-tower history tensor', unit="task", leave=False) as task_progress:
        hist_products = torch.tensor(
            hist_products_list,
            dtype=torch.long
        )
        task_progress.update(1)
    with tqdm(total=1, desc='Creating two-tower weight tensor', unit="task", leave=False) as task_progress:
        hist_weights = torch.tensor(
            hist_weights_list,
            dtype=torch.float32
        )
        task_progress.update(1)

    with torch.no_grad():
        with tqdm(total=1, desc='Encoding two-tower customer histories', unit="task", leave=False) as task_progress:
            tower1_output = trained_model.encode_history_tower(
                hist_products,
                hist_weights
            ).numpy()
            task_progress.update(1)

    future_product_indices = [
        stockcode_to_index[product_code]
        for product_code in product_code_to_matrix_index
    ]
    number_of_products = len(product_code_to_matrix_index)
    rank_dtype = np.uint16 if number_of_products <= np.iinfo(np.uint16).max else np.uint32
    two_tower_ranks = np.zeros(
        (len(customer_id_to_matrix_index), len(product_code_to_matrix_index)),
        dtype=rank_dtype,
    )

    for cust_index in tqdm(customer_indices, desc='Scoring two-tower recommendations', unit="customer", leave=False):
        customer_id = customer_id_list[cust_index]
        scores = product_latent @ tower1_output[cust_index]
        ranked_product_indices = np.argsort(-scores[future_product_indices], kind='stable')
        two_tower_ranks[customer_id_to_matrix_index[customer_id], ranked_product_indices] = np.arange(
            1, number_of_products + 1, dtype=rank_dtype,
        )
        two_tower_ranks[customer_id_to_matrix_index[customer_id], ranked_product_indices[1000:]] = number_of_products

    return two_tower_ranks

# ==========================================================
# Preparing training data
# ==========================================================

# Get unique customers
all_customers = df_calib["Customer_ID"].dropna().unique().to_numpy()

# Shuffle customers reproducibly
np.random.seed(42)
np.random.shuffle(all_customers)

# Split customers into two sets
split_point = len(all_customers) // 2


customer_set1 = set(all_customers[:split_point])

# Create calibration training dataset
df_calib1 = df_calib[df_calib["Customer_ID"].isin(customer_set1)].copy()


# Cutoff at 2/3 of the total time span
start_date = df_calib["InvoiceDate"].min()
end_date = df_calib["InvoiceDate"].max()

cutoff_date = start_date + (end_date - start_date) * (2 / 3)

# First 2/3 of the time period
df_history1 = df_calib1[
    df_calib1["InvoiceDate"] <cutoff_date
].copy()

# Last 1/3 of the time period
df_future1 = df_calib1[
    df_calib1["InvoiceDate"] >= cutoff_date
].copy()

(
    trained_model, stockcode_to_index, index_to_stockcode, pad_embed_index
)= train_2Tower(
    df_history1, df_future1, df_catalog_calib, df_catalog_calib, 
    df_catalog_all, EPOCHS = 4, dim_layer_2=700, dim_layer_3=700 
)

# Shared matrix indices: rows are cleaned calibration customers; columns are holdout products.
with tqdm(total=1, desc='Creating score-matrix index maps', unit="task", leave=False) as task_progress:
    matrix_customer_ids = sorted(df_calib['Customer_ID'].dropna().astype('int64').unique().tolist())
    matrix_product_codes = df_catalog_holdout['StockCode'].dropna().astype(str).drop_duplicates().tolist()
    index_to_matrix_customer_id = dict(enumerate(matrix_customer_ids))
    index_to_matrix_product_code = dict(enumerate(matrix_product_codes))
    customer_id_to_matrix_index = {customer_id: index for index, customer_id in index_to_matrix_customer_id.items()}
    product_code_to_matrix_index = {product_code: index for index, product_code in index_to_matrix_product_code.items()}
    index_maps = {
        'index_to_customer_id': index_to_matrix_customer_id,
        'index_to_stockcode': index_to_matrix_product_code,
    }
    task_progress.update(1)

maps_path = 'data/working_data/recommendation_index_maps.pkl'
if os.path.exists(maps_path):
    with open(maps_path, 'rb') as file:
        assert pickle.load(file) == index_maps, 'Existing index maps do not match the current customers and products.'
else:
    with open(maps_path, 'wb') as file:
        pickle.dump(index_maps, file, protocol=pickle.HIGHEST_PROTOCOL)

two_tower_ranks_path = 'data/working_data/two_tower_ranks.parquet'
if not os.path.exists(two_tower_ranks_path):
    two_tower_ranks = recommend_2Tower(
        df_calib,
        df_catalog_holdout,
        df_catalog_all,
        trained_model,
        stockcode_to_index,
        index_to_stockcode,
        pad_embed_index,
        customer_id_to_matrix_index,
        product_code_to_matrix_index,
    )
    with tqdm(total=1, desc='Saving two-tower rank matrix', unit="task", leave=False) as task_progress:
        pd.DataFrame(
            two_tower_ranks,
            columns=[str(index) for index in range(two_tower_ranks.shape[1])],
        ).to_parquet(two_tower_ranks_path, index=False, compression='zstd', use_dictionary=False)
        task_progress.update(1)
    del two_tower_ranks

ranks_path = 'data/working_data/repeat_purchase_ranks.parquet'
if not os.path.exists(ranks_path):
    ranked_repeat, repeat_ranks = repeat_model(df_calib, df_catalog_calib, df_catalog_holdout)
    number_of_products = len(product_code_to_matrix_index)
    maximum_rank = max((len(products) for products in ranked_repeat.values()), default=0)
    rank_dtype = np.uint16 if max(number_of_products, maximum_rank) <= np.iinfo(np.uint16).max else np.uint32
    repeat_purchase_ranks = np.full(
        (len(customer_id_to_matrix_index), number_of_products),
        number_of_products,
        dtype=rank_dtype,
    )
    for customer_id, product_ranks in tqdm(repeat_ranks.items(), desc='Building repeat-purchase rank matrix', unit="customer", leave=False):
        customer_index = customer_id_to_matrix_index[customer_id]
        for product_code, rank in product_ranks.items():
            if product_code in product_code_to_matrix_index:
                repeat_purchase_ranks[customer_index, product_code_to_matrix_index[product_code]] = rank
    with tqdm(total=1, desc='Saving repeat-purchase rank matrix', unit="task", leave=False) as task_progress:
        pd.DataFrame(
            repeat_purchase_ranks,
            columns=[str(index) for index in range(repeat_purchase_ranks.shape[1])],
        ).to_parquet(ranks_path, index=False, compression='zstd')
        task_progress.update(1)
    del repeat_purchase_ranks, ranked_repeat, repeat_ranks
