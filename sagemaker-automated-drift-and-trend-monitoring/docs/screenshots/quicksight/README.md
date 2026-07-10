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
QUICKSIGHT_IDENTITY_REGION=us-east-1
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

The dashboard shows:
- **Model performance metrics**: Accuracy, precision, recall, F1, ROC-AUC
- **Drift detection trends**: Data drift and model drift over time
- **Inference metrics**: Volume and latency
- **Feature-level drift scores**: Per-feature drift analysis

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

**Next Steps**: Once the dashboard is created, you can access it anytime at:
- QuickSight Console → **Dashboards** → **"Fraud Detection Governance Dashboard"**
