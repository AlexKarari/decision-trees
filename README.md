# Decision Tree Classifier 
---

## Problem Statement

Telco loses approximately 26% of its customer base annually to churn. Each lost customer represents a gap in recurring revenue that costs 5–7× more to replace than to retain. The goal is to predict which customers are likely to leave before they do, enabling proactive retention offers.

This project builds a **Decision Tree classifier entirely from NumPy** — no sklearn for the model itself — to understand how trees make decisions at the algorithm level.

---

## Algorithm: Decision Tree (CART)

A decision tree learns a hierarchy of yes/no rules from data. At each node, it asks: **which feature and threshold best separates churners from non-churners?** It measures "best" using **Gini Impurity**.

### Gini Impurity

$$\text{Gini}(S) = 1 - \sum_{k} p_k^2$$

- **Gini = 0.0** → perfectly pure node (all same class)
- **Gini = 0.5** → maximally mixed (50/50 split, binary)
- **Our root node** starts at Gini ≈ 0.385 (26% churn rate)

### Information Gain

$$\text{Gain} = \text{Gini(parent)} - \left[\frac{|S_L|}{|S|} \cdot \text{Gini}(S_L) + \frac{|S_R|}{|S|} \cdot \text{Gini}(S_R)\right]$$

The tree greedily picks the split with the highest Information Gain at every node.

### Stopping Criteria (Preventing Overfitting)

| Parameter | Value | Effect |
|-----------|-------|--------|
| `max_depth` | 8 | Tree can ask at most 8 questions in sequence |
| `min_samples_split` | 20 | Nodes with < 20 samples become leaves |
| `min_samples_leaf` | 10 | Each child must have ≥ 10 samples |

---

## Dataset

**Telco Customer Churn** — [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

| Property | Value |
|----------|-------|
| Customers | 7,043 |
| Features | ~20 (after encoding: 26) |
| Churn rate | 26.5% |
| Target | Binary — `Churn` (0=stay, 1=leave) |

**Key features:** tenure, MonthlyCharges, TotalCharges, Contract type, InternetService type, PaymentMethod, and 15+ service add-ons.

---

## Results

| Metric | Train | Test |
|--------|-------|------|
| Accuracy | 0.8269 | 0.7793 |
| Precision | 0.6816 | 0.5854 |
| Recall | 0.6528 | 0.5775 |
| F1-Score | 0.6669 | 0.5814 |


### sklearn Validation

The scratch implementation was validated against `sklearn.tree.DecisionTreeClassifier` with identical hyperparameters. Minor metric differences are expected due to tie-breaking in split selection; the algorithm and results are equivalent.

---

## Evaluation Metrics: Why These, and What They Mean for Business

### Why not just Accuracy?

Our dataset has 74% non-churners. A model that **always predicts "stay"** achieves 74% accuracy while catching **zero churners**. Accuracy hides this failure completely on imbalanced datasets.

### The Four Metrics in Plain English

| Metric | Plain English | Business Impact |
|--------|---------------|-----------------|
| **Accuracy** | "Out of every 100 predictions, how many were correct?" | Misleading here — a useless model scores 74% |
| **Precision** | "Of all customers we flagged as churners, what fraction actually churned?" | Low precision = wasted retention budget on non-churners |
| **Recall** | "Of all actual churners, what fraction did we catch?" | Low recall = missed churners walking out the door |
| **F1-Score** | "A single number balancing precision and recall" | Use when comparing models; penalises extreme imbalances |

### Our Priority Metric: Recall

For churn prediction, **Recall is the primary metric**. A missed churner (false negative) costs far more than a wasted retention offer (false positive):

- Average customer LTV in telecom: ~$1,200
- Cost of a retention offer: ~$50–100
- Missing 100 churners/month: $120,000 in lost LTV
- 100 wasted offers: $5,000–10,000

The asymmetry is clear. We tolerate some false alarms to ensure we catch most churners.

### The Precision–Recall Tradeoff

Raising the classification threshold (from 0.5 toward 1.0) increases Precision but decreases Recall. Lowering it does the opposite. The right threshold depends on the business cost structure 

---

## Feature Importance

Decision trees rank features by how much each one contributed to reducing impurity across all splits, weighted by the number of samples at each node.

**Top predictors of churn (run the notebook to see yours):**

| Rank | Feature | Business Interpretation |
|------|---------|-------------------------|
| 1 | Contract type | Month-to-month customers have no switching cost |
| 2 | tenure | New customers haven't built loyalty yet |
| 3 | InternetService type | Fiber optic users may face reliability issues |
| 4 | MonthlyCharges | High bills increase price sensitivity |

These features directly inform retention strategy: prioritise outreach to month-to-month, short-tenure, fiber optic customers.

---

## Decision Tree vs. Logistic Regression

| Dimension | Decision Tree | Logistic Regression |
|-----------|---------------|---------------------|
| Interpretability | ✅ Printable decision rules | ⚠️ Coefficients require interpretation |
| Non-linear patterns | ✅ Handles naturally | ❌ Needs feature engineering |
| Feature interactions | ✅ Captured automatically | ❌ Needs manual terms |
| Feature scaling | ✅ Not required | ⚠️ Required |
| Overfitting risk | ⚠️ High without constraints | ✅ Lower |
| Probability calibration | ⚠️ Leaf frequencies only | ✅ Well-calibrated |
| Best for | Explainability, non-linear data | Probability scoring, large datasets |

---

## Visualizations Generated

| File | Description |
|------|-------------|
| `gini_impurity_curve.png` | Gini as a function of class proportion |
| `confusion_matrix.png` | TP/TN/FP/FN with business labels |
| `feature_importance.png` | Top features ranked by Gini reduction |
| `depth_vs_performance.png` | Overfitting: train vs test as depth increases |
| `dt_vs_lr_comparison.png` | Side-by-side comparison with Logisitc Regression model |

---

## Quick Start

```bash
# 1. Clone and navigate
git clone https://github.com/AlexKarari/decision-trees.git
cd ml-learning/decision-trees

# 2. Install dependencies
pip install -r requirements.txt

# 3. Add the dataset
# Download from: https://www.kaggle.com/datasets/blastchar/telco-customer-churn
# Place at: data/WA_Fn-UseC_-Telco-Customer-Churn.csv

# 4. Train the model
python train_model.py

# 5. Explore the notebook
jupyter notebook notebooks/learning.ipynb
```

**Expected output from `train_model.py`:**
```
Loaded 7,043 rows × 21 columns
Churn rate: 26.5%  (1,869 churners / 7,043 total)
Features after encoding: 26
Train: 5,634 samples  |  Test: 1,409 samples

Training Decision Tree from scratch...
Tree built — actual depth: 8, total nodes: ...

Scratch Decision Tree (Test):
  Accuracy    : ~0.78
  Precision   : ~0.59
  Recall      : ~0.58
  F1-Score    : ~0.58
```

---

## Project Structure

decision-trees/
├── notebooks/
│   └── learning.ipynb    ← 19 cells: theory → implementation → comparison
├── src/
│   ├── __init__.py
│   └── decision_tree.py        ← DecisionTreeScratch + Node (pure NumPy)
├── data/
│   └── .gitkeep                ← CSV excluded via .gitignore
├── models/
│   └── .gitkeep                ← .pkl excluded via .gitignore
├── visualisations/
│   └── *.png                   ← Generated by notebook
├── train_model.py              ← Production training script
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Key Learnings

1. **Gini Impurity** is a fast, effective purity measure — equivalent to entropy in practice but cheaper to compute
2. **Greedy splitting** means trees don't backtrack — each split is locally optimal, not globally
3. **Feature importance** is a by-product of the tree structure, not a post-hoc calculation
4. **No scaling needed** — unlike gradient-based algorithms, trees split on raw values
5. **Overfitting is the main risk** — an unconstrained tree memorises training data perfectly
6. **Recall > Accuracy** for imbalanced classification — the confusion matrix tells the real story
