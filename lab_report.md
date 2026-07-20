# Lab Report: Linear Regression through Gradient Descent

**Course:** Machine Learning Lab  
**Dataset:** UCI Student Performance (Portuguese Language) — `student-por.csv`  
**Language:** Python 3  

---

## Aim

To implement **Linear Regression from scratch using the Gradient Descent optimization algorithm** and apply it to the UCI Student Performance (Portuguese) dataset to predict student final grades (`G3`). We also compare the effect of different learning rates on model convergence and evaluate each model using standard regression metrics.

---

## Objectives

1. Load and explore the `student-por.csv` dataset.
2. Perform Exploratory Data Analysis (EDA) and inspect dataset summary statistics.
3. Preprocess data — handle missing values and encode categorical features using one-hot encoding (`drop_first=True`).
4. Select features (`X`: all columns except `G3`) and target (`y`: `G3`).
5. Split data into training (80%) and testing (20%) sets.
6. Standardize features using `StandardScaler`.
7. Implement Gradient Descent Linear Regression **from scratch** (without `sklearn.LinearRegression`).
8. Train models with learning rates: `0.001`, `0.01`, `0.1` for **1000 iterations**.
9. Plot Cost vs Iterations for all learning rates on a single graph.
10. Evaluate all models using MAE, MSE, RMSE, and R² Score.
11. Identify the best learning rate and plot Actual vs Predicted values.
12. Plot Residual Errors for the best model.
13. Output sample predictions comparing Actual vs Predicted values.

---

## Dataset Description

The **UCI Student Performance Dataset** contains information about students in secondary education collected from two Portuguese schools. We use the **Portuguese language** (`student-por.csv`) subset which contains **649 student records** and **33 attributes**.

| Attribute | Description |
|-----------|-------------|
| `school` | Student's school (GP: Gabriel Pereira, MS: Mousinho da Silveira) |
| `sex` | Student's gender (F: female, M: male) |
| `age` | Student's age (15–22) |
| `address` | Urban (U) or Rural (R) |
| `famsize` | Family size (LE3: ≤3, GT3: >3) |
| `Pstatus` | Parent cohabitation status (T: together, A: apart) |
| `Medu` | Mother's education level (0: none, 1: primary, 2: 5th-9th grade, 3: secondary, 4: higher) |
| `Fedu` | Father's education level (0: none, 1: primary, 2: 5th-9th grade, 3: secondary, 4: higher) |
| `Mjob` | Mother's job (teacher, health, services, at_home, other) |
| `Fjob` | Father's job (teacher, health, services, at_home, other) |
| `reason` | Reason to choose this school (home, reputation, course, other) |
| `guardian` | Student's guardian (mother, father, other) |
| `traveltime` | Home to school travel time (1: <15 min, 2: 15-30 min, 3: 30 min to 1 hr, 4: >1 hr) |
| `studytime` | Weekly study time (1: <2 hrs, 2: 2-5 hrs, 3: 5-10 hrs, 4: >10 hrs) |
| `failures` | Number of past class failures (n if 1≤n<3, else 4) |
| `schoolsup` | Extra educational support (yes/no) |
| `famsup` | Family educational support (yes/no) |
| `paid` | Extra paid classes within subject (yes/no) |
| `activities` | Extracurricular activities (yes/no) |
| `nursery` | Attended nursery school (yes/no) |
| `higher` | Wants to take higher education (yes/no) |
| `internet` | Internet access at home (yes/no) |
| `romantic` | With a romantic relationship (yes/no) |
| `famrel` | Quality of family relationships (1: very bad to 5: excellent) |
| `freetime` | Free time after school (1: very low to 5: very high) |
| `goout` | Going out with friends (1: very low to 5: very high) |
| `Dalc` | Workday alcohol consumption (1: very low to 5: very high) |
| `Walc` | Weekend alcohol consumption (1: very low to 5: very high) |
| `health` | Current health status (1: very bad to 5: very good) |
| `absences` | Number of school absences (0 to 93) |
| `G1` | First period grade (0 to 20) |
| `G2` | Second period grade (0 to 20) |
| `G3` | **Final grade (0 to 20) — TARGET VARIABLE** |

**Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Student+Performance)

---

## Dataset Information & Summary Statistics

- **Dimensions:** 649 rows (samples) × 33 columns (features)
- **Data Types:** 16 integer features (`age`, `Medu`, `Fedu`, `traveltime`, `studytime`, `failures`, `famrel`, `freetime`, `goout`, `Dalc`, `Walc`, `health`, `absences`, `G1`, `G2`, `G3`) and 17 string/object categorical features.
- **Descriptive Statistics:**
  - `df.describe()` highlights key distributional properties of numeric columns (e.g., mean G3 is ~11.91 with a std of ~3.23).
  - `df.describe(include="all")` extends this summary to categorical variables, detailing unique category counts, top (most frequent) values, and frequencies.

---

## Theory & Mathematical Formulation

### 1. Linear Hypothesis
Linear Regression predicts the output as a weighted sum of input features plus a bias term:

$$\hat{y} = \mathbf{X} \cdot \boldsymbol{\theta} + b$$

### 2. Cost Function (Mean Squared Error)
The Mean Squared Error (MSE) cost function measures prediction error across all $m$ samples:

$$J(\boldsymbol{\theta}, b) = \frac{1}{2m} \sum_{i=1}^{m} \left( h_\theta(x^{(i)}) - y^{(i)} \right)^2$$

### 3. Gradient Descent Optimization
Gradient Descent minimizes $J(\boldsymbol{\theta}, b)$ by computing partial derivatives with respect to weights $\boldsymbol{\theta}$ and bias $b$:

$$\frac{\partial J}{\partial \boldsymbol{\theta}} = \frac{1}{m} \mathbf{X}^T (\hat{y} - y)$$

$$\frac{\partial J}{\partial b} = \frac{1}{m} \sum_{i=1}^{m} (\hat{y}^{(i)} - y^{(i)})$$

### 4. Parameter Update Rules
Parameters are updated iteratively in the direction of steepest descent:

$$\boldsymbol{\theta} := \boldsymbol{\theta} - \alpha \cdot \frac{\partial J}{\partial \boldsymbol{\theta}}$$

$$b := b - \alpha \cdot \frac{\partial J}{\partial b}$$

where $\alpha$ represents the **learning rate**.

---

## Preprocessing & Feature Scaling Explanation

### 1. Categorical Encoding
Object columns are converted using One-Hot Encoding (`pd.get_dummies(drop_first=True)`), expanding the feature space from 33 to 42 columns (41 feature columns $X$ + 1 target $y$).

### 2. Feature Scaling Necessity (`StandardScaler`)
Features operate on vastly different scales (e.g., `absences` ranges 0–75 while `studytime` ranges 1–4). Unscaled features lead to elongated cost function contours, causing gradient descent updates to oscillate wildly or converge extremely slowly.

`StandardScaler` standardizes each feature:

$$z = \frac{x - \mu}{\sigma}$$

This transforms feature contours into circular topologies, allowing equal contribution across dimensions and enabling faster, more direct convergence.

---

## Training & Stopping Criteria

- **Iterations:** A fixed count of **1000 iterations** was chosen to compare learning rate trajectories fairly across equal step counts.
- **Learning Rates Tested:** $\alpha \in \{0.001, 0.01, 0.1\}$
- **Convergence Tracking:** Cost history is logged after every iteration to inspect convergence curves visually.

---

## Results & Evaluation

### Model Performance Summary Table (Test Set)

| Learning Rate ($\alpha$) | MAE | MSE | RMSE | R² Score | Final Training Cost |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 0.001 | 4.3374 | 19.9695 | 4.4687 | -1.0478 | 10.4588 |
| 0.01 | 0.7904 | 1.5446 | 1.2428 | 0.8416 | 0.7551 |
| **0.10 (Best)** | **0.7651** | **1.4759** | **1.2149** | **0.8487** | **0.7456** |

### Sample Predictions (First 10 Test Instances, Best Model $\alpha = 0.1$)

| Sample # | Actual G3 | Predicted G3 | Error (Actual - Predicted) | Absolute Error |
|:---:|:---:|:---:|:---:|:---:|
| 1 | 19 | 18.40 | +0.60 | 0.60 |
| 2 | 12 | 11.83 | +0.17 | 0.17 |
| 3 | 18 | 18.56 | -0.56 | 0.56 |
| 4 | 11 | 10.81 | +0.19 | 0.19 |
| 5 | 11 | 11.74 | -0.74 | 0.74 |
| 6 | 17 | 16.52 | +0.48 | 0.48 |
| 7 | 18 | 17.69 | +0.31 | 0.31 |
| 8 | 8 | 9.21 | -1.21 | 1.21 |
| 9 | 10 | 10.99 | -0.99 | 0.99 |
| 10 | 11 | 10.53 | +0.47 | 0.47 |

---

## Learning Rate Comparison & Discussion

1. **Fastest Convergence:** $\alpha = 0.1$ achieved the fastest cost drop and lowest final training cost ($0.7456$).
2. **Best Accuracy:** $\alpha = 0.1$ yielded the best test set accuracy ($R^2 = 0.8487$, $RMSE = 1.2149$).
3. **Trade-offs:**
   - Extremely small learning rates ($\alpha = 0.001$) converge too slowly, leaving the model underfitted after 1000 iterations ($R^2 < 0$).
   - Moderately large rates ($\alpha = 0.1$) perform best when paired with standardized features.
   - Excessively large learning rates risk overshooting the minimum or diverging completely.

---

## Conclusion

1. **Implementation Success:** Linear Regression via Gradient Descent was built entirely from scratch using basic matrix operations in NumPy.
2. **Gradient Descent Mechanism:** Convex cost surface guarantees convergence to the global minimum when proper learning rates are selected.
3. **Impact of Scaling:** Feature scaling via `StandardScaler` was essential to prevent feature scale imbalance and allow rapid convergence.
4. **Predictive Capability:** The model achieved an $R^2$ of **0.8487** ($84.87\%$ variance explained) with an RMSE of **1.2149 grade points** on the test set.
5. **Key Feature Influence:** Previous period grades (`G2` and `G1`) exhibited the highest feature weights, making them the primary drivers of final student performance (`G3`).

---

## References

1. **UCI Machine Learning Repository** — Student Performance Dataset: [https://archive.ics.uci.edu/ml/datasets/Student+Performance](https://archive.ics.uci.edu/ml/datasets/Student+Performance)
2. **Scikit-learn Documentation:** [https://scikit-learn.org/stable/](https://scikit-learn.org/stable/)
3. **NumPy Documentation:** [https://numpy.org/doc/](https://numpy.org/doc/)
4. **Pandas Documentation:** [https://pandas.pydata.org/docs/](https://pandas.pydata.org/docs/)
5. **Matplotlib Documentation:** [https://matplotlib.org/stable/](https://matplotlib.org/stable/)
