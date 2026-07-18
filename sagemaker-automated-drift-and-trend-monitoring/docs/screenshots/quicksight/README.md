# QuickSight Setup Guide

This guide shows you how to set up Amazon QuickSight for the MLOps Governance Dashboard in a new AWS account.

---

## Prerequisites

- ✅ CloudFormation stack deployed
- ✅ Athena database with `fraud_detection` tables populated
- ✅ AWS account with admin access

---

## Setup Steps

### Step 1: Sign Up for QuickSight

1. Go to AWS Console → **Amazon QuickSight**
2. Click **"Sign up for QuickSight"**
3. Choose **Enterprise Edition** (required for programmatic dashboard creation)
4. Configure:
   - **Authentication**: IAM Identity Center or IAM only
   - **Account name**: Choose any name (e.g., `fraud-detection-monitoring`)
   - **Email**: Your admin email
5. Click **"Finish"** and wait 2-3 minutes

![QuickSight Sign Up](./Mlops-1.png)

---

### Step 2: Configure Regions in .env

QuickSight supports different regions for identity (where you signed up) and data (where your S3/Athena data lives).

Add to your `.env` file:

```bash
# The region where your data lives (S3, Athena)
AWS_DEFAULT_REGION=us-west-2

# The region where you signed up for QuickSight
QUICKSIGHT_IDENTITY_REGION=us-west-2
```

**Note**: If your data is in us-east-1, both values can be `us-east-1`. If you signed up for QuickSight in a different region, update `QUICKSIGHT_IDENTITY_REGION` accordingly.

These values are also configurable in `src/config/config.yaml`.

---

### Step 3: Run the Governance Dashboard Notebook

1. Open `notebooks/4_governance_dashboard.ipynb`
2. **Restart kernel** (to reload `.env` with the QuickSight region)
3. **Run all cells**

The notebook will create:
- Athena data source
- 5 datasets (inference, accuracy, drift, features, monitoring)
- Dashboard with visualizations

![Creating Data Source](./Mlops-2.png)

![Creating Datasets](./Mlops-3.png)

---

## Final Result

After running all notebooks (1-4), your QuickSight dashboard will look like this:

![QuickSight Dashboard](./Mlops-4.png)

> **Where do these numbers come from?** Every drift and classification metric in the dashboard is produced by **Evidently AI** inside the scheduled drift Lambda (`fraud-detection-drift-monitor`). Evidently's output is written to the `monitoring_responses` Athena table; QuickSight reads that table directly. There is no separate drift computation in the SQL or in QuickSight — the same numbers are logged to MLflow on the same run, so the dashboard and any MLflow experiment view of the same `monitoring_run_id` are guaranteed to agree.

The dashboard shows:
- **Model performance metrics** (from Evidently `ClassificationPreset`): Accuracy, precision, recall, F1, ROC-AUC
- **Drift detection trends** (from Evidently `DataDriftPreset` and `ClassificationPreset`): Data drift and model drift over time
- **Inference metrics** (from raw `inference_responses` Athena table, not Evidently): Volume and latency
- **Feature-level drift magnitudes** (from Evidently `DataDriftPreset`, normalized to `drift_magnitude` in `evidently_reports.py`): Per-feature drift analysis — visualized as `drift_magnitude` (× past threshold; 1.0 = at threshold, ≥ 3.0 = severe). Test-agnostic across features regardless of which statistical test Evidently picked per column (KS, Chi-square, Wasserstein, or Jensen-Shannon)

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| "QuickSight not subscribed" | Complete Step 1 (sign up) |
| "Access Denied" | Ensure your user has QuickSight Admin role |
| "Directory information not found" | Verify `QUICKSIGHT_IDENTITY_REGION` matches where you signed up |
| Dashboard shows "No data" | Verify Athena tables have data: `SELECT COUNT(*) FROM fraud_detection.inference_responses` |

---

## Cost

- **QuickSight Enterprise**: $24/user/month
- **Additional viewers** (read-only): $5/month (capped)

---

## Creating Custom Visualizations with Natural Language

QuickSight Q allows you to create custom charts by asking questions in plain English. This is especially powerful for ad-hoc drift analysis and trend exploration beyond the pre-built 30-visual dashboard.

![Build Visuals with Natural Language](./BuildVisualsWithNaturalLanguage.png)

### How to Use QuickSight Q

1. Open your QuickSight dashboard
2. Click the **Q search bar** at the top
3. Type your question in natural language
4. QuickSight automatically generates the appropriate visualization
5. Click **"Add to dashboard"** to save the visual

### Sample Natural Language Queries for Drift Analysis

#### **Drift Trend Analysis**

**Top drifting features over time:**
```
Show me a time series chart of the top 5 drifted features over the last 30 days 
with drift_magnitude on the y-axis, grouped by day. Include a horizontal line at 
magnitude=1.0 (the drift threshold). Add a second chart below showing daily 
prediction volume. Highlight days where more than 3 features had magnitude > 1.0 in red.
```

**What it generates:**
- Dual-axis chart with feature drift trends and volume correlation
- Threshold reference line to identify violations
- Color-coded alerts for high-drift days

---

**Feature drift severity distribution:**
```
Create a heatmap showing which features drifted each day over the last 14 days. 
Features on rows, dates on columns, color intensity by drift_magnitude. Highlight 
any cell with drift_magnitude > 3.0 in dark red (severe drift — 3× past threshold).
```

**What it generates:**
- Grid view of feature × date drift intensity
- Quick identification of chronic drifters vs. one-off spikes

---

**Drift score evolution by model version:**
```
Show drift_magnitude trends comparing model version 1 vs version 2 over the last 
7 days. Use separate lines for each version. Add a trend line to show if 
drift is improving or worsening after retraining.
```

**What it generates:**
- Multi-series comparison to validate retraining effectiveness
- Trend analysis showing drift trajectory

---

#### **Model Performance Analysis**

**ROC-AUC degradation timeline:**
```
Create a line chart showing ROC-AUC over the last 30 days with the baseline 
ROC-AUC as a horizontal reference line. Color the line green when above 
baseline, red when below. Add data labels on the 5 lowest points.
```

**What it generates:**
- Visual performance tracking with automatic baseline comparison
- Instant identification of worst-performing days

---

**Prediction distribution shift (Sankey diagram):**
```
Create a Sankey diagram showing how prediction bucket distributions shifted 
from the baseline week to last week. Show flows from training data buckets 
(very_low to very_high fraud probability) to current production buckets. 
Highlight any bucket that changed by more than 10 percentage points in red.
```

**What it generates:**
- Flow visualization of prediction score migration
- Identifies concept drift (score distribution shifts)
- Red highlighting for significant bucket changes (>10pp)

**Example use case:** If your "high confidence fraud" bucket (0.8-1.0 probability) was 5% of training predictions but is now 15% in production, the Sankey will show a thick red flow, indicating the model is predicting fraud more aggressively.

---

**Ground truth coverage and accuracy:**
```
Show a combo chart with ground truth coverage percentage (bars) and model 
accuracy (line) over the last 30 days. Add a warning annotation on days 
where coverage drops below 20%.
```

**What it generates:**
- Dual metric visualization showing data quality vs. performance
- Alerts when model drift metrics become unreliable (low coverage)

---

#### **Feature-Level Deep Dives**

**Credit limit drift investigation:**
```
Show me the credit_limit feature distribution from training data vs last 7 days 
of production data as overlapping histograms. Include mean, median, and standard 
deviation for both distributions. Highlight bins where production frequency is 
more than 2x training frequency in orange.
```

**What it generates:**
- Side-by-side distribution comparison
- Statistical summary to quantify shift magnitude
- Outlier bin identification

---

**Repeat offender features:**
```
Create a bar chart showing how many times each feature has been flagged for drift 
in the last 30 days. Sort descending. Color bars green for 0-2 flags, yellow for 
3-5 flags, red for 6+ flags. Add a table below listing retraining recommendations 
for features with 6+ flags.
```

**What it generates:**
- Chronic drift ranking (retraining priority list)
- Color-coded severity scoring
- Actionable retraining guidance

---

#### **Correlation and Anomaly Detection**

**Drift vs. volume correlation:**
```
Create a scatter plot with daily prediction volume on x-axis and drift percentage 
on y-axis for the last 60 days. Add a trend line. Highlight outliers where volume 
is high but drift is normal (green) or volume is low but drift is high (red).
```

**What it generates:**
- Identifies spurious drift caused by low sample sizes
- Finds genuine drift despite high volumes (true alerts)

---

**Cross-model drift comparison:**
```
Show a grouped bar chart comparing average drift scores across all deployed model 
versions over the last 14 days. Group by model version, color by severity 
(low/medium/high drift). Add a line showing inference volume per version.
```

**What it generates:**
- Multi-model drift landscape
- Version comparison for rollback decisions

---

### Tips for Writing Effective Queries

1. **Be specific about time ranges**: "last 30 days" vs. "last week" vs. "since Jan 1"
2. **Specify chart type**: line chart, bar chart, heatmap, Sankey, scatter plot
3. **Define thresholds explicitly using `drift_magnitude`**: ">1.0 magnitude (drifted)", ">3.0 magnitude (severe)", ">10 percentage points AUC drop", ">20% of features drifted"
4. **Request color coding**: "highlight in red", "color green when above baseline"
5. **Ask for multiple visuals**: "Add a second chart below", "with a table underneath"
6. **Include statistical summaries**: "with mean and median", "add trend line"

### When to Use Natural Language Queries vs. Pre-Built Dashboard

| Use Natural Language Q | Use Pre-Built Dashboard |
|------------------------|-------------------------|
| Ad-hoc investigation of specific features | Daily/weekly monitoring routine |
| Custom time ranges (last 14 days, specific date range) | Standard 7/30-day lookback windows |
| Comparing multiple model versions side-by-side | Tracking single deployed model |
| Creating one-off reports for stakeholders | Ongoing governance and compliance |
| Exploring correlations not in the 30 visuals | Established drift/performance metrics |
| Testing "what-if" threshold changes | Production alerting at configured thresholds |

---

**Next Steps**: Once the dashboard is created, you can access it anytime at:
- QuickSight Console → **Dashboards** → **"Fraud Detection Governance Dashboard"**
- Use **QuickSight Q** (search bar) for custom natural language queries
