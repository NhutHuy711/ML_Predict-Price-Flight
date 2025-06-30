# ML-Predict-Price-Flight
This project conducts an end-to-end analysis and modeling process on a flight pricing dataset. The goal is to understand influential factors in pricing and build predictive models for estimating ticket prices based on multiple flight attributes.

##  Dataset

The dataset (`Clean_Dataset.csv`) contains flight-related features such as:
- **Airline**
- **Source and Destination Cities**
- **Departure and Arrival Time**
- **Class** (Economy/Business)
- **Duration and Total Stops**
- **Price** (Target Variable)

##  1. Data Preprocessing

- Unnecessary columns (`Unnamed: 0`, `flight`) were removed.
- Checked for and dropped duplicates.
- Categorical features were inspected and prepared for encoding.
- Missing values were assessed and handled appropriately.

##  2. Exploratory Data Analysis (EDA)

Extensive visualizations were employed:
- **Pie charts** to analyze airline, source, and destination city distributions.
- **Boxplots** to visualize how features like `airline`, `city`, `class`, and `stops` influence `price`.
- **Histograms** to observe the distribution of numerical variables such as `duration`.

### Notable Insights:
- **Vistara** and **Air India** operate the majority of flights and exhibit broader price ranges.
- **Business class** flights predictably cost more than **Economy**.
- **More stops** and **longer durations** tend to increase ticket price.

##  3. Feature Engineering

- Time-related fields were analyzed and transformed.
- Categorical features were encoded.
- Features with high cardinality or low relevance were removed.

##  4. Model Development

Three regression models were developed and evaluated:
- **Linear Regression**
- **Decision Tree Regressor**
- **Random Forest Regressor**

### Model Evaluation:
- Performance metrics include **MAE**, **RMSE**, and **R² Score**.
- Applied **5-Fold Cross-Validation** to ensure robust evaluation.


