# ML-Predict-Price-Flight
This project presents a comprehensive exploratory data analysis (EDA) on a cleaned airline dataset, with the aim of identifying patterns and factors that influence flight pricing. The study employs statistical summaries and various visualizations to examine relationships among categorical and numerical features.

Dataset Overview
The dataset used in this analysis is titled Clean_Dataset.csv, and includes key attributes such as:

airline: Name of the carrier

source_city and destination_city: Departure and arrival locations

departure_time and arrival_time: Temporal information of the flights

duration, total_stops, and class: Operational characteristics

price: Target variable for analysis

Environment Setup
The following Python libraries are used for data manipulation and visualization:

python
Copy
Edit
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
🔧 Data Preparation
The dataset is imported and initial inspections are performed (e.g., shape, null values).

Non-informative columns such as Unnamed: 0 and flight are removed due to high cardinality and lack of predictive value.

Duplicate records are identified and addressed.

The data is preserved through a working copy to ensure integrity during transformations.

Exploratory Data Analysis (EDA)
1. Airline Distribution and Price Range
Distribution of flights across different airlines is visualized using pie charts.

A boxplot is used to analyze the variation in ticket prices among airlines.

Key Insight: Vistara and Air India account for the majority of flights and also exhibit a broader price range.

2. Source and Destination Cities
Flight frequency and price variability are analyzed based on both source and destination cities.

Results are illustrated via pie charts and boxplots.

Observation: Specific city pairs exhibit notable price differences.

3. Other Factors (available in full notebook)
The effect of class, total_stops, duration, and departure/arrival time on price is also explored.

Correlation analysis is conducted to identify multicollinearity or strong linear relationships.

Visualizations
The study employs various visualization techniques to support findings:

Pie charts: Distribution across categorical variables (e.g., airlines, cities)

Boxplots: Price dispersion by categorical groups

Count plots and histograms: Frequency analysis

Preliminary Findings
Airline, class, and total stops are strong indicators of ticket pricing.

Business class flights predictably incur significantly higher fares.

Duration and stopovers demonstrate a positive correlation with price.

Data Cleaning and Feature Engineering
Categorical variables are encoded for future modeling.

Time-based features are transformed and extracted for improved utility.

Redundant or noisy variables are removed to reduce dimensionality.

Future Work
Implementation of predictive models (e.g., Linear Regression, Random Forest) using the cleaned dataset.

Model evaluation using standard regression metrics: MAE, RMSE, and R².

Hyperparameter tuning and cross-validation for improved generalizability.
