# Criteo Uplift Modelling

An end-to-end uplift modelling analysis using the [Criteo Uplift Dataset](https://ailab.criteo.com/ressources/criteo-uplift-modeling-dataset/) — a large-scale randomised controlled experiment across 13 million users from a real digital advertising campaign.

The project covers the full analytical cycle: experiment validation, A/B test analysis, and individual-level uplift modelling, with an emphasis on business interpretation alongside technical implementation.

---

## The Business Problem

When a company runs a marketing campaign, it wants to know whether the campaign *caused* customers to respond — or whether those customers would have converted anyway. Targeting users who would have converted regardless wastes budget. Targeting users who respond negatively to contact ("Sleeping Dogs") actively destroys value.

**Uplift modelling** addresses this by predicting the *incremental effect* of a treatment on each individual — identifying who is more likely to respond *because* of the campaign, not despite it or independently of it.

---

## Project Structure

```
criteo-uplift-analysis/
├── README.md
├── data/
│   └── .gitkeep           # Dataset not committed — see below
└── notebooks/
    ├── 01_eda_experiment_validity.ipynb
    ├── 02_ab_test_analysis.ipynb
    └── 03_uplift_modelling.ipynb
```

### Notebook 1: EDA & Experiment Validity
- Dataset overview and structure
- Outcome and treatment distributions
- Exposure analysis: the distinction between being *targeted* and being *reached*
- Feature distributions and correlation
- Randomisation check using Standardised Mean Difference (SMD) and Love plot

### Notebook 2: A/B Test Analysis
- Hypothesis definition and significance threshold
- Average Treatment Effect (ATE) — absolute and relative lift
- Effect size (Cohen's h)
- Two-proportion z-test with p-values and conclusions
- 95% confidence intervals, visualised
- Intent-to-Treat (ITT) vs Average Treatment Effect on the Treated (ATT)
- Retrospective power analysis

### Notebook 3: Uplift Modelling
- The four uplift segments: Persuadables, Sure Things, Lost Causes, Sleeping Dogs
- Class Variable Transformation (CVT) approach
- Train/test split with stratification; resampling applied to training set only
- Three models compared: Logistic Regression, Random Forest, XGBoost
- Qini curve evaluation and model comparison
- Uplift score distribution and targeting efficiency
- Second model using `exposure` as treatment variable

---

## Dataset

The dataset is the **Criteo Uplift v2.1** dataset, available from the [Criteo AI Lab](https://ailab.criteo.com/ressources/criteo-uplift-modeling-dataset/).

Download `criteo-uplift-v2.1.csv` (~3.25 GB) and place it in the `data/` folder before running the notebooks. The file is not committed to this repository due to its size.

The dataset was originally explored via [this Kaggle notebook](https://www.kaggle.com/code/hughhuyton/criteo-uplift-modelling).

| Field | Description |
|---|---|
| f0 – f11 | Twelve anonymised numerical user features |
| treatment | Randomly assigned to campaign (1) or control (0) |
| exposure | Whether the user actually saw the advertisement (1/0) |
| visit | Whether the user visited the advertiser's website — primary outcome |
| conversion | Whether the user made a purchase — secondary outcome |

---

## Key Findings

**Experiment validity**
The randomisation was successful — Standardised Mean Differences across all 12 features are well within the |SMD| < 0.1 threshold, confirming that outcome differences between groups can be attributed to the treatment rather than pre-existing user differences.

**A/B test**
The campaign had a statistically significant positive effect on both visit rate (+1.03pp, 27% relative lift) and purchase rate (+0.12pp, 59% relative lift). With 13 million users the experiment was massively overpowered — effect size and commercial significance matter more than p-values here.

**The exposure problem**
Only 3.6% of treated users were actually exposed to the advertisement. The ITT/ATT ratio is 36x for visits and 45x for purchases — the headline ATE substantially understates the campaign's true effectiveness when the ad actually reaches someone. The most impactful lever available is improving ad delivery and reach, not campaign creative or targeting refinement.

**Uplift models**
All three models outperform random targeting, but their Qini scores are essentially indistinguishable: Random Forest (0.0098), XGBoost (0.0096), Logistic Regression (0.0095). The additional complexity of XGBoost produces no material gain on this dataset. For a production deployment, Logistic Regression would be the rational choice — simpler, faster, and more interpretable.

**The standout finding: the exposure model**
A second uplift model trained with `exposure` as the treatment variable achieves a Qini score of 0.3108 — a 30x improvement over the treatment model. The user features contain strong predictive signal about who will respond to the advertisement *once reached*, but very weak signal about who will happen to see it. The features describe user characteristics; they do not describe whether an ad will be served and viewed — which is driven by platform mechanics beyond the user's profile.

---

## Requirements

```
pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost
scipy
statsmodels
```

Install with:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost scipy statsmodels
```

---

## Methodology Notes

**Why CVT rather than a Two-Model approach?**
The Class Variable Transformation (Kane et al., 2014) reformulates uplift as a standard multiclass classification problem, making it straightforward to apply any off-the-shelf classifier. The Two-Model (T-Learner) approach — training separate models on treatment and control groups and subtracting predictions — is a natural alternative and would be worth exploring for comparison.

**Why Qini rather than AUUC?**
Both the Qini curve and the Area Under the Uplift Curve (AUUC) are standard evaluation metrics for uplift models. The Qini curve is used here because it is the more established convention in the incrementality testing literature and is directly comparable across datasets when normalised.

**CVT and class imbalance**
The CVT formula produces a structural bias toward positive uplift scores when outcome rates are very low — as they are in this dataset (visit rate ~5%, purchase rate ~0.3%). The model correctly *ranks* users by persuadability, but the absolute score values should not be interpreted as probabilities of positive uplift, and the Sleeping Dog segment does not materialise in practice. This is an honest limitation of the approach on heavily imbalanced outcome data.