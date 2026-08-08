# Loom Video Walkthrough Script (2-3 Minutes)

A structured, section-by-section script you can follow while recording your Loom video.

---

### **[0:00 – 0:20] Introduction**
> *"Hi, I'm [Your Name]. This is my walkthrough of the Spotter Freight Rate Prediction challenge. My goal was to train a machine learning model to predict freight rates for a future validation set covering November and December 2025, and to forecast daily rates for a fixed Lexington-to-Fort-Wayne route in December."*

*(Visual: Show the repository folder or the top of `freight_rate_prediction.ipynb`)*

---

### **[0:20 – 0:50] Dataset Overview & Temporal Boundaries**
> *"The training data spans January to October 2025 — 48,000 loads. The validation set is November to December 2025 — 12,000 loads. Because the validation period is a completely future window, I verified this date range first thing in the notebook. This confirmed that a standard random cross-validation split would cause data leakage, so I used a time-based split instead: training on January to August, and validating locally on September to October."*

*(Visual: Scroll to the Date Range Check cell and its output)*

---

### **[0:50 – 1:20] Key Data Exploration Findings**
> *"I ran several key explorations. First, the target variable `posted_rate` is heavily right-skewed — ranging from $57 to over $25,500 — so I applied a log transform for training. The correlation matrix confirmed `distance` is the dominant predictor with a 0.91 correlation. However, the other features like `market_index`, `quote_signal`, and `weight` have low linear correlations but carry important non-linear signals.*
>
> *I also visualized the day-of-week pricing patterns, which clearly showed that rates peak mid-week when carrier demand is highest, and I plotted the market index and quote signal over time to confirm they are genuinely dynamic signals and not constants. I also checked the route volume distribution and rate ranges by distance bin to understand which routes and distances the model would be most confident about."*

*(Visual: Scroll through the boxplot, correlation heatmap, day-of-week barplot, and market index time series cells)*

---

### **[1:20 – 1:45] Data Quality & Imputation**
> *"I found two data quality issues: 300 missing weight values in training and 374 missing market index values. For weights, I plotted the weight distribution per equipment type — Dry Van, Reefer, and Flatbed each have very different profiles. So instead of a global median fill, I imputed missing weights using the equipment-group median, which is much more accurate. Market index was filled with the training median."*

*(Visual: Scroll to the weight distribution by equipment type cell)*

---

### **[1:45 – 2:10] Feature Engineering**
> *"The most important feature engineering decision was cyclic date encoding. November and December are fully absent from training, so encoding the raw month as a category would produce broken all-zero vectors at inference time. Instead, I encoded `day_of_year` as sine and cosine — this lets the model learn smooth, continuous annual seasonality that generalizes cleanly to any future date.*
>
> *I also computed the Haversine great-circle distance from coordinates as an additional spatial feature, and created a `days_since_start` time trend to capture market inflation over the year."*

*(Visual: Highlight the preprocess_and_engineer function in the notebook or train_predict.py)*

---

### **[2:10 – 2:35] Modeling & Results**
> *"I used LightGBM, which natively handles the categorical route features, is robust to outliers, and doesn't need feature normalization. On the local September–October validation set, the model achieved an R-squared of 0.82, a MAPE of 6.6%, and an MAE of just $149. The actual-vs-predicted scatter plot confirmed no systematic bias. The feature importance chart confirmed that distance dominates, but temporal and spatial engineered features also contribute meaningfully."*

*(Visual: Scroll to the validation metrics output, then to the feature importance plot, then the actual vs predicted scatter)*

---

### **[2:35 – 3:00] December Predictions & Submission**
> *"Finally, I retrained on all 10 months of data and generated predictions for the full 12,000 validation loads and the 31 December daily scenarios. For December, I sourced the daily market conditions from the December portion of the validation set. All outputs were verified using the official `score.py` scorer, which confirmed correct formatting and generated the December trend chart.*
>
> *All the code is clean, documented, and reproducible. Thank you!"*

*(Visual: Show the December chart output and then the validated prediction CSV)*
