# Dataset

`creditcard.csv.gz` is the full dataset, gzip-compressed (66 MB; 144 MB once uncompressed). pandas reads it directly, no need to unzip it:

```python
import pandas as pd
df = pd.read_csv("data/creditcard.csv.gz")
```

**Source:** [Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) — ULB Machine Learning Group, on Kaggle.

- 284,807 transactions made by European cardholders over two days in September 2013
- 492 frauds (0.173%): the classes are extremely imbalanced
- 30 features: `Time`, `Amount`, and `V1`–`V28` (PCA-transformed, anonymised) + the `Class` label

Please check the dataset's licence on its Kaggle page before redistributing it.

> Do not add the uncompressed `creditcard.csv` to Git: it exceeds GitHub's 100 MB limit (it is listed in `.gitignore`).
