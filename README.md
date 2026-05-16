
# Comment Category Prediction Challenge: Imbalanced Multiclass Classification

**Final Rank:** 388 / 2744 (Top 15%) | **Metric:** Macro F1-Score | **Final Score:** 0.8627

## Project Overview
This repository details a machine learning pipeline developed for the Comment Category Prediction Challenge. The dataset provides insight into how an online platform processes and categorizes user-generated comments. Each record represents a single comment and includes metadata such as interaction feedback, symbolic expressions, topic reference indicators, and internal system signals.

Our task was to explore this dataset and build predictive models to accurately determine the final category assigned to each comment by uncovering patterns across textual, numerical, and categorical features. The primary technical hurdle was extreme class imbalance, where the critical minority class (Class 3) represented a fraction of the total dataset, necessitating a strict optimization for the Macro F1 metric.

## Technical Architecture & Feature Engineering
To capture the behavioral nuances of different user classes, the pipeline expanded beyond standard TF-IDF vectorization:

* **Temporal Anomaly Mapping:** Utilized Kernel Density Estimation (KDE) to isolate class-specific temporal patterns, engineering binary flags for specific behavioral windows (e.g., the "Witching Hour" and "Midday Trough").
* **Interaction Ratios:** Engineered features to quantify "Controversy" by mathematically modeling the ratio of downvotes to total engagement volume.
* **Sparsity Handling:** Binarized and aggregated highly sparse features (such as symbolic expressions and specific emoticon usage) to create clean, split-friendly variables for the tree algorithms.
* **Logarithmic Transformations:** Applied `np.log1p` to heavily right-skewed internal system signals to normalize distributions and prevent binning issues.

## Modeling Strategy: The Evolution
The modeling approach went through several iterations as we evaluated how different algorithms reacted to the 14,000+ sparse TF-IDF columns. We tested three primary models: Logistic Regression, XGBoost, and LightGBM.

### 1. The Tuning Process (Broad to Surgical)
We initially cast a wide net using `RandomizedSearchCV` to map the general hyperparameter space. However, as cross-validation became computationally prohibitive, we pivoted to curated manual tuning. We designed specific architectural configurations (e.g., "Deep & Slow" vs. "Fast & Regularized") and evaluated them strictly on an internal holdout set, allowing us to safely lock in the highest-performing configuration.

### 2. Why LightGBM Dominated
While Logistic Regression was tested, it struggled to capture the complex, non-linear relationships in the text interactions and often became overconfident. XGBoost provided a strong baseline, but LightGBM ultimately served as our main predictive engine for three specific reasons:
1. **Leaf-Wise Growth:** Unlike XGBoost's depth-wise approach, LightGBM grows trees leaf-wise. This proved incredibly efficient for highly sparse text matrices, allowing it to find complex word combinations faster.
2. **Native Imbalance Handling:** The `class_weight='balanced'` parameter, combined with a high `min_child_samples` threshold, acted as a mathematical brake pedal. It forced the model to hunt for the rare Class 3 without memorizing noisy typos.
3. **GPU Memory Efficiency:** Working with 14,000+ columns often triggers Out-Of-Memory (OOM) errors. LightGBM's histogram binning was drastically lighter on memory compared to XGBoost, allowing us to utilize a much higher number of estimators.

### 3. Final Submission Strategy
Instead of ensembling different algorithms (which risked smoothing out our delicate LightGBM probabilities), the final submission relied on a highly tuned, standalone LightGBM model. We trusted the robust architecture to generalize to the hidden Private Leaderboard, avoiding last-minute probability threshold hacks that often lead to overfitting.

## Future Work
If this pipeline were to be scaled or deployed in a production environment, further optimizations would include:
* **Deeper Exploratory Data Analysis (EDA):** Conducting a more granular EDA on the raw text—such as analyzing vocabulary overlap between sibling classes or time-series sentiment shifts—to engineer even stronger discriminative features.
* **Confusion Matrix-Driven Tuning:** While we utilized normalized confusion matrices for visual validation during training, a future iteration would build an automated pipeline to dynamically adjust class weights based on specific misclassification rates, further tuning the boundaries between the most challenging classes.

