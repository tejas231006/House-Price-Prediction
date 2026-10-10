# House Price Prediction Using Machine Learning

## Final project
**House Price Prediction Using Machine Learning: A Comparative Analysis of Regression Models**

### Practical coverage
- Practical 1: Data cleaning and preprocessing
- Practical 3: Linear Regression and GridSearchCV
- Practical 4: KNN
- Practical 5: Decision Tree / Random Forest and hyperparameter tuning

### Models
1. Linear Regression
2. KNN Regressor
3. Decision Tree Regressor
4. Random Forest Regressor

### Metrics
- MAE: lower is better
- RMSE: lower is better
- R²: higher is better

## Dataset
Download the Kaggle dataset: **House Sales in King County, USA**

- Dataset link: https://www.kaggle.com/datasets/harlfoxem/housesalesprediction
- Place the downloaded `train.csv` file in:

```bash
data/train.csv
```

The dataset is intentionally not included in this repository.

## Run

Install dependencies:

    pip install -r requirements.txt

Train:

    python train.py

Open notebook:

    jupyter notebook

Run UI:

    streamlit run app/app.py

## Output
Training creates:
- models/best_model.pkl
- outputs/results/model_comparison.csv
- outputs/figures/*.png

## Final workflow
Data Cleaning → Preprocessing → Train/Test Split → 4 Models → GridSearchCV
→ Metrics → Comparison → Best Model → Streamlit UI → Deployment
