# Corporate-Fatigue-Risk-Modeling


This project predicts burnout symptoms in workers using machine learning models. It identifies the strongest predictors of burnout, examines potential biases related to gender and remote work, and simulates a policy intervention targeting top predictors.


## **Exploratory Data Analysis (EDA)**  

Before modeling, we conducted **exploratory data analysis (EDA)** to understand the distribution of features and relationships between variables.

### **Numeric Variables**  
- Most **numeric variables exhibited uniform distributions**, suggesting **no strong skewness or natural clustering**.  
- This indicates that **burnout symptoms do not correlate with specific numeric thresholds**, making direct predictions based on individual numeric features challenging.
<p align="center">
 <img src="https://github.com/RoryQo/Corporate-Fatigue-Risk-Modeling/blob/main/Visualizations/Dist.png" 
       alt="Distribution of Numeric Variables" width="600px"> 
</p>

### **Categorical Variables**  
- We examined **categorical feature relationships** using **Cramér's V** and **correlation heatmaps**.  
- The results showed **low correlations between categorical variables**, meaning **no single category strongly influenced another**.  
- **Cramer V tests** between categorical variables and the target variable (**burnout symptoms**) **showed little correlation**, suggesting that achieving **high model accuracy might be difficult** due to weak direct relationships.  

<div align="center">

| **Feature**         | **Cramér's V** |
|---------------------|---------------|
| Gender             | 0.0053        |
| Marital Status     | 0.0024        |
| Job Role          | 0.0022        |
| Health Issues      | 0.0023        |
| Company Size       | 0.0000        |
| Department         | 0.0000        |
| Location          | 0.0000        |

</div>

These findings **suggest that burnout prediction requires more than just simple linear associations** and that models will need to **capture non-linear relationships and complex interactions between features** to improve performance.
**Note:** These are initial findings before data wrangling, like feature scaling and dummy variable creation

## **Methodology**  

### **Data Processing**  
Before modeling, the dataset was cleaned and preprocessed:  
- Workers who reported **occasional burnout symptoms** were **encoded as "Yes"**, allowing for **binary classification**.
   Additionally, indicating health issues was encoded to yes to create binary classification, saving overall data set dimensionality when we encode the rest of the categorical variables with dummies.

```
# For binary encoding include the occasional symptoms to the yes category
# Reducing Dimensionality
df["Burnout_Symptoms"] = df["Burnout_Symptoms"].replace("Occasional", "Yes")

df["Health_Issues"] = df["Health_Issues"].replace("Both", "Yes")
df["Health_Issues"] = df["Health_Issues"].replace("Physical", "Yes")
df["Health_Issues"] = df["Health_Issues"].replace("Mental", "Yes")
```

- Not having **health issues** was originally coded as **"na"**, we will replace with **"No"** for future modeling processes.

- Use label encoding for company size, instead of dummies to reduce dimensionality and keep it as an ordinal variable
```
# Label encode bc size is ordinal 
#Initialize LabelEncoder
label_encoder = LabelEncoder()

# Encode a specific column (e.g., "Category_Column")
df["Company_Size"] = label_encoder.fit_transform(df["Company_Size"])
```

  
- **One-hot encoding (dummy variables)** was applied to all remaining categorical variables.  


### **Model Creation**  
We trained and evaluated multiple machine learning models:  
- **k-Nearest Neighbors (KNN)**
- **Naïve Bayes (NB)**
- **Support Vector Machine (SVM)**
- **Random Forest**
- **XGBoost**
- **Neural Networks**

Each model was optimized using **grid search** for hyperparameter tuning and **5-fold cross-validation** to ensure robustness on unseen data.

- Example code with the neural network
```
# Define hyperparameters to tune
param_grid = {
    'hidden_layer_sizes': [(64,), (128,)],  # Different layer architectures
    'activation': ['relu'],  # Try different activation functions
    'solver': ['adam'],  # Optimizers: Adam vs. SGD
    'alpha': [0.0001],  # Regularization strength
    'learning_rate': ['constant', ]  # Learning rate strategies
}

# Initialize MLP Classifier
mlp = MLPClassifier(max_iter=200, random_state=42)

# Grid Search with Cross-Validation (5-fold)
grid_search = GridSearchCV(mlp, param_grid, cv=5, scoring='f1_weighted', n_jobs=-1, verbose=1)

# Fit the model on training data
grid_search.fit(X_train, y_train)
```

### **Model Evaluation**  
Initial testing with **Naïve Bayes and SVM** revealed that accuracy alone was not a good evaluation metric. Since burnout symptoms are less frequent in the dataset, a model focusing too heavily on the majority class would appear to perform well in terms of accuracy but would fail to correctly predict those who do experience burnout symptoms. This imbalance led us to focus on the weighted F1 score, which better accounts for both precision and recall across classes.
- **Example:** Naïve Bayes exclusively predicted "Burnout" cases.  

To address this, we **shifted our focus to the weighted F1-score**, which balances precision and recall across classes.

### **Model Selection & Computational Time**  
- **Initially, we tested simple models** like **KNN** and **Naïve Bayes**, assuming that **more complex models would perform better**.
- However, **despite extensive hyperparameter tuning, models like SVM, Random Forest, XGBoost, and Neural Networks did not outperform KNN**. Some performed even worse.

<p align="center">
  <img src="https://github.com/RoryQo/Corporate-Fatigue-Risk-Modeling/blob/main/Visualizations/Modelf1comp.jpg" alt="Model Performance Comparison" width="600px">
</p>

- **Computational time was also a major factor.**  
  - While all models had reasonable runtimes, **Neural Networks and SVM were significantly more expensive computationally**.
  - **KNN proved to be the simplest, fastest, and most effective model**, making it the best choice for predicting burnout.

<p align="center">
  <img src="https://github.com/RoryQo/Corporate-Fatigue-Risk-Modeling/blob/main/Visualizations/ModelCompTimes.jpg" alt="Model Computation Time Comparison" width="600px">
</p>


## **Results**  

### **Key Features**  
To retrieve feature importance results, we used permutation-based feature importance to measure how much each predictor contributes to burnout predictions. This method involves randomly shuffling a single feature's values while keeping all others intact, then observing the resulting drop in model performance. To ensure robustness, we applied 5-fold cross-validation when computing permutation importance scores, ensuring that feature contributions remained consistent across different subsets of data. The XGBoost model has a built-in feature importance, which ranks features based on how much they improve decision tree splits. An alternative that will be considered in the future is absolute SHAP values, which is a scoring method to retreive factor importance from all types of models.

```
# Compute permutation importance
result = permutation_importance(best_knn, X_test, y_test, scoring='f1_weighted', n_repeats=5, random_state=42)

# Get feature importances from XGBoost
feature_importances = best_xgboost.feature_importances_
```

The most important predictors of burnout symptoms in the **KNN model** included:  
- **Marital Status (Widowed)**
- **Department (Sales, Marketing, HR)**
- **Location**
- **Years of Experience**
- **Job Satisfaction**

  <img src="https://github.com/RoryQo/Corporate-Fatigue-Risk-Modeling/blob/main/Visualizations/knnimp.png" 
       alt="KNN Feature Importance" width="600px">


**Neural Networks & XGBoost independently confirmed these findings** while also identifying:  
- **Physical Activity Levels**  
- **Health Issues**

Interestingly, **Remote Work and Gender were not significant predictors of burnout**.  
However, the **Neural Network model found that experiencing gender discrimination was the single most important predictor of burnout**.

<img src="https://github.com/RoryQo/Corporate-Fatigue-Risk-Modeling/blob/main/Visualizations/MLPImp.png" 
       alt="MLP Feature Importance" width="450px">
<img src="https://github.com/RoryQo/Corporate-Fatigue-Risk-Modeling/blob/main/Visualizations/xgimport.jpg" 
       alt="XGBoost Feature Importance" width="450px">
  

### **Bias in Gender and Remote Work**  
To check for bias, we analyzed **average values of the top 10 burnout predictors by gender**.  
- No significant differences were found, **suggesting gender was not a direct factor in predicting burnout symptoms**.

<p align="center">
  <img src="https://github.com/RoryQo/Corporate-Fatigue-Risk-Modeling/blob/main/Visualizations/Gender.jpg" 
       alt="Gender Analysis" width="600px">
</p>
  
- A similar analysis was conducted for **remote work status**, which also showed no meaningful differences.  

<p align="center">
  <img src="https://github.com/RoryQo/Corporate-Fatigue-Risk-Modeling/blob/main/Visualizations/Remotework.jpg" 
       alt="Gender Analysis" width="600px">
</p>

This suggests that **gender and remote work do not directly or indirectly influence burnout risk**.

## **Future Policy Effect Simulation**  

If the company could design a policy that reduces the 2 strongest predictors of burnout by twenty percent, how would this affect your prediction? How effective would this policy be? These factors will be taken from the SHAP values test.

**`1.`** **Simulate a 20% Reduction in These Predictors**
Gender Discrimination: A workplace policy aimed at reducing discrimination, improving inclusivity, and enforcing strict anti-discrimination measures.
Job Satisfaction: Initiatives such as better compensation, improved work-life balance, career development opportunities, and management support.
To simulate this, we modify the dataset by reducing the values of these features by 20% while ensuring that the changes remain realistic and within the data’s distribution.
**Note:** Since we scaled the data to run neural networks and SVMs all the variables are numeric, and can be reduced this way. & that reducing predictors might not be the correct direction depending on the predictor, i.e. reducing sleep hours would actually increase burnout because the relationship is more sleep is correlated with less burnout.

```
# Apply a 20% reduction for non-binary features
for feature in top_predictors:
    if feature in X_test_policy.columns:
        # Reduce all values by 20% proportionally
        X_test_policy[feature] *= 0.8
        print(f" Reduced {feature} by 20%")
```

**`2.`** **Re-run Burnout Predictions**
Using the modified dataset with reduced burnout risk factors, we re-run predictions with our KNN model and compare the new burnout rate to the original predictions.

```
# Predict burnout before and after policy
y_pred_original = best_knn.predict(X_test)
y_pred_policy = best_knn.predict(X_test_policy)
```

<p align="center">
  <img src="https://github.com/RoryQo/Corporate-Fatigue-Risk-Modeling/blob/main/Visualizations/PolicyEval.png" 
       alt="Policy Evaluation Impact" width="700px">
</p>

**`3.`** **Evaluate Effectiveness**
If the burnout prediction rate decreases significantly, the policy is effective.
If the model still predicts high burnout, other factors may be driving stress in the workplace.
We measure effectiveness using F1-score improvement and relative reduction in predicted burnout cases.

Model Performance Comparison



<p align="center">
  <img src="https://github.com/RoryQo/Corporate-Fatigue-Risk-Modeling/blob/main/Visualizations/Diff.png" 
       alt="Difference in Burnout Prediction" width="500px">
</p>


<div align="center">

<table>
  <tr>
    <th colspan="4"><strong>Original Model Performance</strong></th>
    <th colspan="4"><strong>Policy-Adjusted Model Performance</strong></th>
  </tr>
  <tr>
    <th>Metric</th>
    <th>Precision</th>
    <th>Recall</th>
    <th>F1-Score</th>
    <th>Metric</th>
    <th>Precision</th>
    <th>Recall</th>
    <th>F1-Score</th>
  </tr>
  <tr>
    <td>False</td>
    <td>0.32</td>
    <td>0.20</td>
    <td>0.25</td>
    <td>False</td>
    <td>0.33</td>
    <td>0.21</td>
    <td>0.26</td>
  </tr>
  <tr>
    <td>True</td>
    <td>0.66</td>
    <td>0.78</td>
    <td>0.72</td>
    <td>True</td>
    <td>0.67</td>
    <td>0.79</td>
    <td>0.72</td>
  </tr>
  <tr>
    <td>Accuracy</td>
    <td colspan="3">0.59</td>
    <td>Accuracy</td>
    <td colspan="3">0.59</td>
  </tr>
  <tr>
    <td>Macro Avg</td>
    <td>0.49</td>
    <td>0.49</td>
    <td>0.48</td>
    <td>Macro Avg</td>
    <td>0.50</td>
    <td>0.50</td>
    <td>0.49</td>
  </tr>
  <tr>
    <td>Weighted Avg</td>
    <td>0.55</td>
    <td>0.59</td>
    <td>0.56</td>
    <td>Weighted Avg</td>
    <td>0.55</td>
    <td>0.59</td>
    <td>0.57</td>
  </tr>
  <tr>
    <td colspan="4"><strong>Original Predicted Burnout Rate: 0.7887</strong></td>
    <td colspan="4"><strong>Predicted Burnout Rate After Policy: 0.7864</strong></td>
  </tr>
</table>

</div>


**`4.`** **Conclude**
This minimal effect suggests **burnout is a complex and multifaceted issue that cannot be significantly reduced by only addressing two factors.** A more **holistic approach** is required to meaningfully reduce burnout.

## **Discussion**  

The findings from our models suggest that **burnout prediction remains a challenging task** given the available features. Our selected **KNN model achieved an accuracy in the low 60s**, which indicates **only marginal predictive power**. 

The **low Cramér's V and Spearman correlation scores** further confirm that **most features have little direct predictive influence on burnout symptoms**. This suggests two possibilities:  
1. **The features provided may not be strong predictors of burnout.** Other unmeasured variables (such as workplace culture, managerial support, or mental health history) may play a greater role.
2. **More complex relationships may exist.** There could be **nonlinear interactions** between features, which simple models fail to capture.

Additionally, the **simulated policy intervention, which aimed to reduce the two strongest predictors of burnout, had minimal impact on overall burnout rates**. This further reinforces the idea that **burnout is a multifaceted, complex issue** that cannot be easily addressed by targeting a few individual factors.

### **Future Considerations**  
- Exploring **nonlinear interactions and feature engineering** to capture hidden patterns.
- Incorporating **external data** (such as workplace policies or mental health history) for a richer model.
- Testing **deep learning approaches** that might better capture subtle relationships.

Ultimately, **burnout is not driven by a single factor but rather by a combination of workplace, personal, and societal influences.** Any meaningful intervention will likely require **holistic solutions rather than isolated policy changes.**


