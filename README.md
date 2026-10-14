# Customer Behavior Prediction Case Study

This project builds an end-to-end customer analytics workflow for a UK online
retailer. Starting from raw transaction lines, it prepares calibration and
holdout datasets, cleans noisy retail data, predicts future purchase frequency,
estimates customer lifetime value, and evaluates several product recommendation
strategies.

The business goal is to answer three practical questions:

1. Which customers are likely to remain active, and which customers are likely
   to churn?
2. What is each customer's expected customer lifetime value (CLV)?
3. Which products should be recommended to each customer in the future period?

The analysis uses the UCI Online Retail II dataset, which contains transaction
history from a UK-based online retailer.

## Results

| Task | Holdout result |
| --- | --- |
| Zero-purchase prediction | ROC AUC: **0.748** |
| Aggregate CLV | Predicted **6.86M** vs actual **6.67M** (**+2.94%**) |
| Customer-level CLV | Pearson correlation: **0.913**; WAPE: **51.34%** |
| All-product recommendation | Precision@10: **0.394**; NDCG@10: **0.436** |
| New product recommendation | Precision@10: **0.112**; NDCG@10: **0.120** |
| New-catalogue recommendation | Precision@10: **0.126**; NDCG@10: **0.139** |

The CLV estimates are most useful for ranking and segmenting customers rather than forecasting each customer's exact spend. The recommendation rankers outperform global popularity, including on the more difficult new-product tasks.


## Project Structure

| Notebook | Purpose |
| --- | --- |
| `00_Create_case_study_data.ipynb` | Downloads the raw Online Retail II data, creates a time-based calibration/holdout split, and builds product catalog files. |
| `01_Cleaning.ipynb` | Cleans calibration and holdout transactions, handles returns/cancellations, removes non-product activity, and creates order-level datasets. |
| `02_Purchase_Frequency.ipynb` | Fits a Pareto/NBD model to predict future purchase counts and customer activity. |
| `03_CLV.ipynb` | Fits a Gamma-Gamma monetary-value model and combines it with purchase-frequency predictions to estimate CLV. |
| `04_product_reco.ipynb` | Builds and evaluates product recommendation models, including popularity, repeat purchase, market-basket, semantic, two-tower, and LightGBM ranking approaches. |

## Data Flow

The notebooks should be run in order:

```text
00_Create_case_study_data.ipynb
        |
        v
01_Cleaning.ipynb
        |
        +--> 02_Purchase_Frequency.ipynb
        |           |
        |           v
        |      03_CLV.ipynb
        |
        v
04_product_reco.ipynb
```

## Input Data

The project uses the UCI Online Retail II dataset. The raw file contains
1,067,371 transaction lines with the following fields:

- `Invoice`
- `StockCode`
- `Description`
- `Quantity`
- `InvoiceDate`
- `Price`
- `Customer ID`
- `Country`

The transaction history is split using 5 December 2010 as the cutoff:

- Transactions on or before 5 December 2010 form the calibration period.
- Transactions after 5 December 2010 form the holdout period.

The calibration period is used for cleaning and model fitting. The holdout
period is kept as unseen future data for evaluation.

## Generated Files

The workflow creates the following main files:

| File | Description |
| --- | --- |
| `data/given_data/calibration_raw.csv` | Raw calibration-period transaction lines. |
| `data/given_data/holdout_transaction.csv` | Future-period transaction lines used for holdout evaluation. |
| `data/given_data/prod_catalog_calib.csv` | Product catalog observed during the calibration period. |
| `data/given_data/prod_catalog_holdout.csv` | Product catalog observed during the holdout period. |
| `data/working_data/calib_all_transaction.csv` | Cleaned calibration transaction-level data for recommendation modeling. |
| `data/working_data/calib_order.csv` | Cleaned calibration customer-order-level data. |
| `data/working_data/holdout_order.csv` | Cleaned holdout customer-order-level data. |
| `data/working_data/lifetimes.csv` | Customer-level RFM and Pareto/NBD outputs used for CLV modeling. |

## Cleaning Approach

The cleaning notebook prepares the raw retail data for modeling by:

- Standardizing identifiers, dates, customer IDs, product descriptions, and text
  fields.
- Matching negative-quantity rows to earlier positive sales to handle returns
  and cancellations.
- Removing administrative and non-product activity such as fees, postage,
  discounts, manual adjustments, and bank charges.
- Removing duplicate rows.
- Creating transaction-level data for recommendation modeling.
- Aggregating customer-day orders for purchase-frequency and CLV modeling.

This step converts noisy invoice-line data into stable customer purchase
histories.

## Purchase-Frequency Modeling

`02_Purchase_Frequency.ipynb` uses a Pareto/NBD model because the retailer is a
non-contractual business. Customers do not explicitly cancel, so churn must be
inferred from purchasing behavior rather than observed directly.

The model uses each customer's:

- Purchase frequency
- Recency
- Customer age

It estimates:

- Probability that the customer is still active
- Expected number of purchases in the holdout period
- Risk of zero future purchases

The purchase-frequency model overpredicts total holdout purchase volume, but it
still produces an informative customer ranking. Across 4,253 customers, it
predicts 17,025 purchases compared with 12,493 observed purchases. Customer-level
ranking remains useful, with strong Pearson correlation and moderate Spearman
correlation between predicted and observed purchase counts.

## CLV Modeling

`03_CLV.ipynb` combines:

- Pareto/NBD expected future purchases
- Gamma-Gamma expected average order value

The Gamma-Gamma model estimates each customer's long-run monetary value while
reducing noise in raw average order value, especially for customers with limited
purchase histories. The notebook uses a 50:50 blend of Gamma-Gamma expected
order value and calibration average order value as the final average order value
estimate.

The final CLV model performs best as a ranking and prioritization tool:

| Customer Group | Share of Actual Holdout CLV Captured |
| --- | ---: |
| Top 1% by predicted CLV | 29.35% |
| Top 5% by predicted CLV | 44.76% |
| Top 10% by predicted CLV | 55.22% |
| Top 20% by predicted CLV | 67.53% |

Predicted total CLV is close to actual holdout CLV at the portfolio level, but
individual customer-level predictions remain uncertain. The model is therefore
most useful for retention targeting, customer tiering, and marketing budget
allocation rather than exact per-customer revenue forecasting.

## Customer Lifetime Value Prediction

The CLV model was evaluated using decile analysis by ranking customers according to their predicted lifetime value and comparing the mean predicted CLV with the mean actual CLV within each decile. The model successfully separates lower-value from higher-value customers, with the highest predicted decile capturing customers with substantially greater realized lifetime value. The close agreement between predicted and actual CLV across most deciles indicates that the model effectively captures customer-value ranking.

<p align="center">
  <img src="images/clv_decile_analysis.png" width="850">
</p>

*Figure: Mean actual and predicted customer lifetime value across predicted CLV deciles.*

## Product Recommendation Modeling

`04_product_reco.ipynb` builds several recommendation approaches:

| Model | Description |
| --- | --- |
| Global popularity | Recommends products that are broadly popular across customers. |
| Repeat purchase | Recommends products the customer has purchased before. |
| Market basket | Uses product co-occurrence patterns to recommend items commonly bought together. |
| Content-based semantic model | Uses product descriptions and sentence-transformer embeddings to recommend semantically similar products. |
| Two-tower neural model | Learns customer and product representations for personalized scoring. |
| LightGBM ranker | Combines model scores and features into a supervised ranking model. |

The recommendation evaluation considers three tasks:

- Overall holdout purchase prediction
- New-to-customer product prediction
- New-catalog product prediction

The notebook evaluates recommendations using ranking metrics such as
Precision@k, Recall@k, MAP@k, and NDCG@k.

## Dependencies

The notebooks use Python and common data science libraries, including:

- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `scipy`
- `pymc`
- `pymc-marketing`
- `arviz`
- `sentence-transformers`
- `torch`
- `lightgbm`
- `implicit`
- `mlxtend`
- `tqdm`

Python and Python-package dependencies are managed with **uv**. The project uses
`pyproject.toml` to declare dependencies, `uv.lock` to record resolved versions,
and `.python-version` to select Python 3.12. From the project directory, run
`uv sync` to create `.venv`, download the required Python version if needed, and
install the locked packages. See the [uv documentation](https://docs.astral.sh/uv/concepts/projects/sync/).

System-level prerequisites are **Git**, **uv**, and a **C++ compiler** for normal
PyMC/PyTensor compiled execution. These tools must be installed separately;
`uv.lock` manages Python packages, not operating-system tools.

### Windows Setup

Run these commands in **Windows PowerShell**.

1. **Install Git:**

   ```powershell
   winget install --id Git.Git -e --source winget
   ```

   Close and reopen PowerShell, then verify:

   ```powershell
   git --version
   ```

2. **Install uv:**

   ```powershell
   winget install --id astral-sh.uv -e
   ```

   Close and reopen PowerShell, then verify:

   ```powershell
   uv --version
   ```

3. **Install a C++ compiler for PyTensor:** The Pareto/NBD and Gamma-Gamma
   models use PyMC/PyTensor. The Windows setup tested for this repository uses
   WinLibs GCC/G++:

   ```powershell
   winget install --id BrechtSanders.WinLibs.POSIX.UCRT -e
   ```

   Close and reopen PowerShell, then verify:

   ```powershell
   g++ --version
   ```

4. **Clone the repository:** First navigate to the parent directory where you
   want to keep your GitHub projects, then run:

   ```powershell
   git clone https://github.com/amdirfan88/customer-clv.git
   cd customer-clv
   ```

5. **Reproduce the Python environment:** From inside `customer-clv`, run:

   ```powershell
   uv sync
   ```

   This creates the project's virtual environment and installs the required
   Python version and packages using the project configuration and lockfile.

6. **Start JupyterLab:** From inside `customer-clv`, run:

   ```powershell
   uv run jupyter lab
   ```

### Windows PyTensor Troubleshooting

PyTensor dynamically compiles native modules during model fitting. If an error
contains `An Application Control policy has blocked this file`, Windows Smart
App Control / Application Control may be blocking a generated compiled module.

Review **Windows Security → App & browser control → Smart App Control settings**.
Read the Windows warning before changing the setting: depending on your Windows
configuration, turning Smart App Control off may not be reversible without
resetting or reinstalling Windows. See [Microsoft's Smart App Control FAQ](https://support.microsoft.com/en-us/windows/security/threat-malware-protection/smart-app-control-frequently-asked-questions).
This is troubleshooting guidance, not a required installation step.

### macOS Setup

The project has also been tested on macOS. Ensure Git and
[uv](https://docs.astral.sh/uv/getting-started/installation/) are installed and a
C++ compiler is available. In the tested macOS environment, PyTensor uses
Apple's `clang++`; verify it in Terminal with:

```bash
clang++ --version
```

In Terminal, navigate to your preferred parent directory, then run:

```bash
git clone https://github.com/amdirfan88/customer-clv.git
cd customer-clv
uv sync
uv run jupyter lab
```
Run `uv sync` and `uv run jupyter lab` from inside `customer-clv`.

## How to Run

After completing the setup above, open the notebooks in JupyterLab and run all
cells in each notebook in this order:

1. `00_Create_case_study_data.ipynb`
2. `01_Cleaning.ipynb`
3. `02_Purchase_Frequency.ipynb`
4. `03_CLV.ipynb`
5. `04_product_reco.ipynb`

Keep JupyterLab running from the project directory so relative data paths
resolve correctly. The first notebook downloads the raw dataset and requires
internet access. See [Data Flow](#data-flow) and [Project Structure](#project-structure)
for the notebook dependencies and purposes.

## Key Takeaways

- A time-based calibration/holdout split gives the project a realistic
  future-prediction setup.
- Pareto/NBD is appropriate for non-contractual churn because customer dropout
  is latent.
- CLV predictions are strongest for ranking customers, not for exact
  individual-level revenue estimates.
- A small share of high-ranked customers captures a large share of future value,
  making the CLV model useful for targeting decisions.
- Product recommendation quality should be judged across multiple objectives,
  especially repeat purchases, new-to-customer discovery, and new-catalog
  recommendations.

