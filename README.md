# Diabetes Analysis




## Background



### What is Diabetes

Diabetes is a chronic metabolic disease characterized by elevated levels of blood glucose (blood sugar). This occurs because the body either cannot produce enough insulin or cannot effectively use the insulin it produces. Over time, uncontrolled blood sugar leads to serious damage to the heart, blood vessels, eyes, kidneys, and nerves.


### The Dataset Overview

The Dataset, which was gathered by the CDC, is a dataset containing healthcare statistics and lifestyle information of people, and whether or not they have diabetes, is pre-diabetic, or healthy.

The dataset can be found here:
- https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators



#### Variables of the Dataset

The dataset has a total of 22 (sans ID) columns:

Demographic Columns:
- Sex - Sex of the patient.
- Age - Age of the patient grouped into 13 bins.
  - 18-24 = 1
  - 25-29 = 2
  - 30-34 = 3
  - ...
  - 80+ = 13
- Education - Educational attainment of the patient.
  - 1 = Did not attend/only kindergarten
  - 2 = Grade 1 to 8
  - 3 = Grade 9 to 11
  - 4 = Grade 12 or GED graduate
  - 5 = 1st year to 3rd year
  - 6 = 4 or more years/graduate
- Income - Income category of the patient based on annual income.
  - 1 = Less than 10,000 USD (approx. PHP 500,000)
  - 5 = Less than 35,000 USD
  - 8 = 75,000 USD or more  

Laboratory Test Columns:
- Diabetes_binary - Indicator for diabetes (Target Variable).
- HighBP - Indicator for low/high BP.
- HighChol - Indicator for low/high cholesterol.
- BMI - Body Mass Index.

Survey Columns
- CholCheck - Indicator for having a cholesterol check within the last 5 years.
- Smoker - Indicator for smoking at least 100 packs in a patient's lifetime.
- Stroke - Indicator if a patient has had a stroke.
- HeartDiseaseorAttack - Indicator if a patient has/had coronary heart disease or myocardial infarction.
- PhysActivity - Indicator if the patient has done any physical activity in the last month.
- Fruits - Indicator if the patient consumes any fruit in a day.
- Veggies - Indicator if the patient consumes any vegetable in a day.
- HvyAlcoholConsump - Indicator if the patient is a heavy drinker (Varies on gender).
- AnyHealthcare - Indicator if a patient has any health care insurance/coverage/plans.
- NoDocbcCost - Indicator of a patient's inability to visit a doctor due to cost.
- GenHlth - Personal judgement of a patient's health.
- MentHlth - Personal assessment of a patient's mental health, determined by the number of days they didn't have a good mental health.
- PhysHlth - Personal assessment of a patient's physical health, determined by the number of days they didn't have a good physical health.
- DiffWalk - Personal assessment of a patient indicating if they have difficulty in mobility.

### Objective of the Analysis

This analysis aims to examine diabetes indicators of different patients. By analyzing key factors such as prior health conditions, habits, diet, and other variables. This analysis seeks to dig up any notable insights that can help further understand diabetes.


## Data Collection

Before we begin, let us first import the necessary libraries that we need for this analysis.

We will also be importing the dataset which can be located here:
- https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset


```python
import kagglehub
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

diabetes_dataframe = pd.read_csv(
     "https://raw.githubusercontent.com/"
      "Lanzero225/Diabetes-Analysis/refs/heads/main/diabetes_dataset.csv"
)
```

We can see a sample of the first 5 records below. As you can see, there are a lot of binary features that can be found.


```python
diabetes_dataframe.head()
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Diabetes_binary</th>
      <th>HighBP</th>
      <th>HighChol</th>
      <th>CholCheck</th>
      <th>BMI</th>
      <th>Smoker</th>
      <th>Stroke</th>
      <th>HeartDiseaseorAttack</th>
      <th>PhysActivity</th>
      <th>Fruits</th>
      <th>...</th>
      <th>AnyHealthcare</th>
      <th>NoDocbcCost</th>
      <th>GenHlth</th>
      <th>MentHlth</th>
      <th>PhysHlth</th>
      <th>DiffWalk</th>
      <th>Sex</th>
      <th>Age</th>
      <th>Education</th>
      <th>Income</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>40.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>5.0</td>
      <td>18.0</td>
      <td>15.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>9.0</td>
      <td>4.0</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>25.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>7.0</td>
      <td>6.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>28.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>...</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>5.0</td>
      <td>30.0</td>
      <td>30.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>9.0</td>
      <td>4.0</td>
      <td>8.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>27.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>...</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>11.0</td>
      <td>3.0</td>
      <td>6.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>24.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>...</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>11.0</td>
      <td>5.0</td>
      <td>4.0</td>
    </tr>
  </tbody>
</table>


Looking at the information of the DataFrame, we can see that all of them are under the  float64 datatypes, which is not the intended type for some of the columns.


```python
diabetes_dataframe.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 253680 entries, 0 to 253679
    Data columns (total 22 columns):
     #   Column                Non-Null Count   Dtype  
    ---  ------                --------------   -----  
     0   Diabetes_binary       253680 non-null  float64
     1   HighBP                253680 non-null  float64
     2   HighChol              253680 non-null  float64
     3   CholCheck             253680 non-null  float64
     4   BMI                   253680 non-null  float64
     5   Smoker                253680 non-null  float64
     6   Stroke                253680 non-null  float64
     7   HeartDiseaseorAttack  253680 non-null  float64
     8   PhysActivity          253680 non-null  float64
     9   Fruits                253680 non-null  float64
     10  Veggies               253680 non-null  float64
     11  HvyAlcoholConsump     253680 non-null  float64
     12  AnyHealthcare         253680 non-null  float64
     13  NoDocbcCost           253680 non-null  float64
     14  GenHlth               253680 non-null  float64
     15  MentHlth              253680 non-null  float64
     16  PhysHlth              253680 non-null  float64
     17  DiffWalk              253680 non-null  float64
     18  Sex                   253680 non-null  float64
     19  Age                   253680 non-null  float64
     20  Education             253680 non-null  float64
     21  Income                253680 non-null  float64
    dtypes: float64(22)
    memory usage: 42.6 MB
    

## Data Pre-Processing

We can first begin our process by transforming our dataset into something suitable for analysis.



### Data Cleaning

#### Removing Duplicate Rows

In this first step of Data Cleaning, let us remove records that are duplicate. This doesn't necessarily mean removing those records with the same values, but those with the same ID.


```python
duplicate_rows = diabetes_dataframe[diabetes_dataframe.duplicated(keep=False)]

duplicate_rows.sort_values(by=list(diabetes_dataframe.columns)).head(10)
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Diabetes_binary</th>
      <th>HighBP</th>
      <th>HighChol</th>
      <th>CholCheck</th>
      <th>BMI</th>
      <th>Smoker</th>
      <th>Stroke</th>
      <th>HeartDiseaseorAttack</th>
      <th>PhysActivity</th>
      <th>Fruits</th>
      <th>...</th>
      <th>AnyHealthcare</th>
      <th>NoDocbcCost</th>
      <th>GenHlth</th>
      <th>MentHlth</th>
      <th>PhysHlth</th>
      <th>DiffWalk</th>
      <th>Sex</th>
      <th>Age</th>
      <th>Education</th>
      <th>Income</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>4517</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>17.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>...</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>8.0</td>
      <td>6.0</td>
      <td>8.0</td>
    </tr>
    <tr>
      <th>207307</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>17.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>...</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>8.0</td>
      <td>6.0</td>
      <td>8.0</td>
    </tr>
    <tr>
      <th>42369</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>18.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>...</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>10.0</td>
      <td>6.0</td>
      <td>8.0</td>
    </tr>
    <tr>
      <th>108949</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>18.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>...</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>10.0</td>
      <td>6.0</td>
      <td>8.0</td>
    </tr>
    <tr>
      <th>17475</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>19.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>...</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>4.0</td>
      <td>6.0</td>
      <td>8.0</td>
    </tr>
    <tr>
      <th>80704</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>19.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>...</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>4.0</td>
      <td>6.0</td>
      <td>8.0</td>
    </tr>
    <tr>
      <th>152374</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>19.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>...</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>4.0</td>
      <td>6.0</td>
      <td>8.0</td>
    </tr>
    <tr>
      <th>91414</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>19.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>...</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>6.0</td>
      <td>6.0</td>
      <td>8.0</td>
    </tr>
    <tr>
      <th>238843</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>19.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>...</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>6.0</td>
      <td>6.0</td>
      <td>8.0</td>
    </tr>
    <tr>
      <th>48850</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>19.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>...</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>9.0</td>
      <td>6.0</td>
      <td>8.0</td>
    </tr>
  </tbody>
</table>

As seen above, we see multiple records that share the same value, but they all come from different people (Different ID). In this unique case, we should not drop any duplicate rows, since this is a representation of real world data, and coincidences such as these occur all the time.

#### Data Type Casting

Before we begin, let us first fix the typings of each columns. This is to also save memory when it comes to processing the dataset.


```python
integer_columns = ['PhysHlth', 'MentHlth', 'GenHlth', "Age", "Education", "Income", "BMI"]
binary_columns = diabetes_dataframe.columns.difference(integer_columns)

diabetes_dataframe[integer_columns] = diabetes_dataframe[integer_columns].astype('int8')
diabetes_dataframe[binary_columns] = (
    diabetes_dataframe[binary_columns]
    .apply(lambda col: col.map({0: False, 1: True}))
    .astype("boolean")
)

diabetes_dataframe.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 253680 entries, 0 to 253679
    Data columns (total 22 columns):
     #   Column                Non-Null Count   Dtype  
    ---  ------                --------------   -----  
     0   Diabetes_binary       253680 non-null  boolean
     1   HighBP                253680 non-null  boolean
     2   HighChol              253680 non-null  boolean
     3   CholCheck             253680 non-null  boolean
     4   BMI                   253680 non-null  int8   
     5   Smoker                253680 non-null  boolean
     6   Stroke                253680 non-null  boolean
     7   HeartDiseaseorAttack  253680 non-null  boolean
     8   PhysActivity          253680 non-null  boolean
     9   Fruits                253680 non-null  boolean
     10  Veggies               253680 non-null  boolean
     11  HvyAlcoholConsump     253680 non-null  boolean
     12  AnyHealthcare         253680 non-null  boolean
     13  NoDocbcCost           253680 non-null  boolean
     14  GenHlth               253680 non-null  int8   
     15  MentHlth              253680 non-null  int8   
     16  PhysHlth              253680 non-null  int8   
     17  DiffWalk              253680 non-null  boolean
     18  Sex                   253680 non-null  boolean
     19  Age                   253680 non-null  int8   
     20  Education             253680 non-null  int8   
     21  Income                253680 non-null  int8   
    dtypes: boolean(15), int8(7)
    memory usage: 9.0 MB
    

As seen above, we have went from 42 MB to 9 MB, which will make this entire process more efficient. We can proceed to the next steps of data preparation. Given how all fields have the same number of non-null values to the total records, we do not have to undergo imputation, and as such, we can move on to finding outliers.

#### Outlier Detection

Checking for outliers will be simple at this point. Now that we have converted most of the columns into datatypes, we can simply check the value counts of integer columns.

- PhysHlth, MentHlth - 1 to 30
- GenHlth - 1 to 5
- Age - 1 to 13
- Education - 1 to 6
- Income - 1 to 8


```python
cols_to_check = ['PhysHlth', 'MentHlth', 'GenHlth', 'Age', 'Education', 'Income']
diabetes_dataframe[cols_to_check].hist(figsize=(15, 10), bins=30, edgecolor='black')
plt.suptitle('Distribution and Range Check for Integer Features')
plt.show()
```


    
![png](diabetes_notebook_files/diabetes_notebook_21_0.png)
    


Upon graphing each of the integer columns (sans BMI), we can see that all the records are within the set bounds. There are also no obvious outliers. Most people in terms of health see themselves as very healthy, with a slight peak at 30, indicating some find themselves to be extremely unhealthy.

In terms of age, we also see that most people are within the middle age bracket of around 40-50 years old.

In terms of education, most people are actually those that studied in college, with very few having stopped at high school or below. The majority of the patients are also college graduates.

As for income, we can see that most patients have a relatively solid annual income, with the dataset being mostly people in the middle class.

With that completed, let us look into BMI, which aren't categorized yet. Let's plot a boxplot to see the interquartile range and those records that are way too far from Q3.


```python
sns.boxplot(x=diabetes_dataframe['BMI'])
plt.title('Outlier Check for BMI')
plt.show()
```


    
![png](diabetes_notebook_files/diabetes_notebook_23_0.png)
    



```python
Q1 = diabetes_dataframe['BMI'].quantile(0.25)
Q3 = diabetes_dataframe['BMI'].quantile(0.75)
IQR = Q3 - Q1

outliers = diabetes_dataframe[(diabetes_dataframe['BMI'] < (Q1 - 1.5 * IQR)) | (diabetes_dataframe['BMI'] > (Q3 + 1.5 * IQR))]
print(f"The 4th quartile starts at BMI: {Q3}")
print(f"Number of BMI outliers: {len(outliers)}")
```

    The 4th quartile starts at BMI: 31.0
    Number of BMI outliers: 9847
    

Upon checking the interquartile range, we can observe that there are 9847 outliers in the BMI column. To see this, let us check these columns.


```python
upper_bound = Q3 + (1.5 * IQR)
print(f"Outliers begin at a BMI of: {upper_bound}")

diabetes_dataframe[diabetes_dataframe['BMI'] >= 41.05].head()
```

    Outliers begin at a BMI of: 41.5
    


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Diabetes_binary</th>
      <th>HighBP</th>
      <th>HighChol</th>
      <th>CholCheck</th>
      <th>BMI</th>
      <th>Smoker</th>
      <th>Stroke</th>
      <th>HeartDiseaseorAttack</th>
      <th>PhysActivity</th>
      <th>Fruits</th>
      <th>...</th>
      <th>AnyHealthcare</th>
      <th>NoDocbcCost</th>
      <th>GenHlth</th>
      <th>MentHlth</th>
      <th>PhysHlth</th>
      <th>DiffWalk</th>
      <th>Sex</th>
      <th>Age</th>
      <th>Education</th>
      <th>Income</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>85</th>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>45</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>...</td>
      <td>True</td>
      <td>False</td>
      <td>4</td>
      <td>2</td>
      <td>30</td>
      <td>True</td>
      <td>False</td>
      <td>9</td>
      <td>6</td>
      <td>8</td>
    </tr>
    <tr>
      <th>97</th>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>45</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>...</td>
      <td>True</td>
      <td>False</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>False</td>
      <td>False</td>
      <td>9</td>
      <td>4</td>
      <td>6</td>
    </tr>
    <tr>
      <th>156</th>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>47</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>...</td>
      <td>True</td>
      <td>False</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>True</td>
      <td>False</td>
      <td>11</td>
      <td>6</td>
      <td>5</td>
    </tr>
    <tr>
      <th>188</th>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>43</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>...</td>
      <td>True</td>
      <td>False</td>
      <td>4</td>
      <td>0</td>
      <td>7</td>
      <td>True</td>
      <td>False</td>
      <td>10</td>
      <td>5</td>
      <td>2</td>
    </tr>
    <tr>
      <th>201</th>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>55</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>...</td>
      <td>True</td>
      <td>False</td>
      <td>4</td>
      <td>0</td>
      <td>15</td>
      <td>False</td>
      <td>False</td>
      <td>11</td>
      <td>5</td>
      <td>2</td>
    </tr>
  </tbody>
</table>


We can check the values that are considered outliers, as seen below:


```python
bmi_outliers_df = diabetes_dataframe[diabetes_dataframe['BMI'] >= 41.05].copy()

bin_edges = list(range(41, 105, 5))

bmi_outliers_df['BMI_Group'] = pd.cut(bmi_outliers_df['BMI'], bins=bin_edges)

outlier_distribution = bmi_outliers_df['BMI_Group'].value_counts().sort_index()
print("Distribution of BMI Outliers:")
print(outlier_distribution)
```

    Distribution of BMI Outliers:
    BMI_Group
    (41, 46]     5751
    (46, 51]     2147
    (51, 56]      843
    (56, 61]      309
    (61, 66]      133
    (66, 71]      102
    (71, 76]      132
    (76, 81]      173
    (81, 86]       85
    (86, 91]       93
    (91, 96]       45
    (96, 101]       7
    Name: count, dtype: int64
    


```python
diabetes_dataframe[diabetes_dataframe['BMI'] >= 96].head()
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Diabetes_binary</th>
      <th>HighBP</th>
      <th>HighChol</th>
      <th>CholCheck</th>
      <th>BMI</th>
      <th>Smoker</th>
      <th>Stroke</th>
      <th>HeartDiseaseorAttack</th>
      <th>PhysActivity</th>
      <th>Fruits</th>
      <th>...</th>
      <th>AnyHealthcare</th>
      <th>NoDocbcCost</th>
      <th>GenHlth</th>
      <th>MentHlth</th>
      <th>PhysHlth</th>
      <th>DiffWalk</th>
      <th>Sex</th>
      <th>Age</th>
      <th>Education</th>
      <th>Income</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>36324</th>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>96</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>...</td>
      <td>True</td>
      <td>False</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>False</td>
      <td>False</td>
      <td>6</td>
      <td>4</td>
      <td>8</td>
    </tr>
    <tr>
      <th>76370</th>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>98</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>...</td>
      <td>True</td>
      <td>False</td>
      <td>2</td>
      <td>10</td>
      <td>0</td>
      <td>False</td>
      <td>False</td>
      <td>3</td>
      <td>4</td>
      <td>6</td>
    </tr>
    <tr>
      <th>76394</th>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>98</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>...</td>
      <td>True</td>
      <td>False</td>
      <td>5</td>
      <td>30</td>
      <td>30</td>
      <td>True</td>
      <td>False</td>
      <td>8</td>
      <td>6</td>
      <td>5</td>
    </tr>
    <tr>
      <th>76396</th>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>98</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>...</td>
      <td>True</td>
      <td>False</td>
      <td>4</td>
      <td>15</td>
      <td>10</td>
      <td>True</td>
      <td>False</td>
      <td>11</td>
      <td>5</td>
      <td>6</td>
    </tr>
    <tr>
      <th>76532</th>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>98</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>...</td>
      <td>True</td>
      <td>False</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>False</td>
      <td>False</td>
      <td>13</td>
      <td>5</td>
      <td>8</td>
    </tr>
  </tbody>
</table>



We are at a standstill. The issue here is that a BMI around 75+ is a bit of a problem, since it looks out of the ordinary and feels like it's only a special case.

We have approximately 500 records where the BMI is above 76. Upon checking, a BMI in that range is feasible, but extremely rare. What we can do is keep them for now as they do not go beyond the human limits

See: https://www.bbc.com/news/av/world-latin-america-42581917

### Feature Engineering



#### BMI Categorization

One of the most common ways BMI can be referred to is with their categorical classification:
- Underweight - Less than 18.5
- Healthy - 18.5 - 24.9
- Overweight - 25 - 29.9
- Obesity - 30+
  - Class 1 Obesity - 30 - 34.9
  - Class 2 Obesity - 35 - 39.9
  - Class 3 Obesity - 40+

For this, we will bin the BMI into a categorical column.

https://www.cdc.gov/bmi/adult-calculator/bmi-categories.html


```python
bmi_bins = [0, 18.5, 24.9, 29.9, 34.9, 39.9, float('inf')]

#bmi_labels = ['Underweight', 'Normal', 'Overweight', 'Class 1 Obese', 'Class 2 Obese', 'Class 3 Obese']
bmi_labels = [1, 2, 3, 4, 5, 6]
diabetes_dataframe['BMI_Category'] = pd.cut(diabetes_dataframe['BMI'], bins=bmi_bins, labels=bmi_labels)
diabetes_dataframe['BMI_Category'].value_counts()
```





<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
    </tr>
    <tr>
      <th>BMI_Category</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>3</th>
      <td>93749</td>
    </tr>
    <tr>
      <th>2</th>
      <td>68953</td>
    </tr>
    <tr>
      <th>4</th>
      <td>53451</td>
    </tr>
    <tr>
      <th>5</th>
      <td>20663</td>
    </tr>
    <tr>
      <th>6</th>
      <td>13737</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3127</td>
    </tr>
  </tbody>
</table>



#### Lifestyle Score

Next, we can try to combine and aggregate all the healthy habits, combining them into a single metric. For this, we can call it 'Healthy_Score'.


```python
diabetes_dataframe['Healthy_Score'] = (
    diabetes_dataframe['PhysActivity'].astype(int) +
    diabetes_dataframe['Fruits'].astype(int) +
    diabetes_dataframe['Veggies'].astype(int) -
    diabetes_dataframe['Smoker'].astype(int) -
    diabetes_dataframe['HvyAlcoholConsump'].astype(int)
)
```

#### Health Distress

Lastly, I will try to combine both bad mental and physical health days to see total bad health days within the last month.


```python
diabetes_dataframe['Total_Bad_Health_Days'] = diabetes_dataframe['MentHlth'] + diabetes_dataframe['PhysHlth']
```

### Initial Correlation Analysis


```python
correlations = diabetes_dataframe.corr()['Diabetes_binary'].sort_values(ascending=False)

plt.figure(figsize=(10, 8))
correlations.drop('Diabetes_binary').plot(kind='barh', color='skyblue')

plt.title('Correlation of Features with Diabetes_binary')
plt.xlabel('Correlation Coefficient')
plt.ylabel('Features')
plt.axvline(x=0, color='black', linestyle='--', linewidth=1) # Reference line at 0
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.show()
```


    
![png](diabetes_notebook_files/diabetes_notebook_39_0.png)
    


Upon creating a correlation graph, plotting the other fields with their correlation with Diabetes_binary, we can see that Income, Education, and PhysActivity have a relatively large negative correlation with the target variable.

Meanwhile, in the positive correlation side, GenHlth, HighBP, DiffWalk, BMI, HighChol, Age, HeartDiseaseOrAttack, and PhysHlth have a relative high correlation.

This means that those with a negative correlation have an inverse relation with the target variable, but those with a postiive correlation have a direct relation.

### Data preparation

#### Train-Test Split

Before we proceed with scaling or balancing, we must first split our data. This is to prevent having class balancing and feature scaling be affected by the test set.


```python
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

bool_cols = diabetes_dataframe.select_dtypes(include=['boolean']).columns
diabetes_dataframe[bool_cols] = diabetes_dataframe[bool_cols].astype('int8')


X = diabetes_dataframe.drop('Diabetes_binary', axis=1)
y = diabetes_dataframe['Diabetes_binary']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
```

#### Handling Class Imbalance


```python
diabetes_dataframe['Diabetes_binary'].value_counts()
```




<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
    </tr>
    <tr>
      <th>Diabetes_binary</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>218334</td>
    </tr>
    <tr>
      <th>1</th>
      <td>35346</td>
    </tr>
  </tbody>
</table>


Upon checking the number of classes for those with diabetes and those without, we can see a huge class imbalance. Around 15% of the target variable are True, while the rest are False. This may result in a model that might lean towards predicting "No Diabetes" everytime.


```python
sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

print(f"Original training shape: {y_train.value_counts()}")
print(f"Resampled training shape: {y_train_res.value_counts()}")
```

    Original training shape: Diabetes_binary
    0    174667
    1     28277
    Name: count, dtype: int64
    Resampled training shape: Diabetes_binary
    0    174667
    1    174667
    Name: count, dtype: int64
    

To handle this class balance, one of the most popular technique is to use Synthetic Minority Over-sampling Technique (SMOTE).

This generates synthetic data based on the minority class, for this case, the True class, by interpolating samples around the minor category.

This lets us create the same number of samples for both classes.



```python
bmi_bins = [0, 18.5, 24.9, 29.9, 34.9, 39.9, float('inf')]

#bmi_labels = ['Underweight', 'Normal', 'Overweight', 'Class 1 Obese', 'Class 2 Obese', 'Class 3 Obese']
bmi_labels = [1, 2, 3, 4, 5, 6]
diabetes_dataframe['BMI_Category'] = pd.cut(diabetes_dataframe['BMI'], bins=bmi_bins, labels=bmi_labels)
diabetes_dataframe['BMI_Category'].value_counts()
```





<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
    </tr>
    <tr>
      <th>BMI_Category</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>3</th>
      <td>93749</td>
    </tr>
    <tr>
      <th>2</th>
      <td>68953</td>
    </tr>
    <tr>
      <th>4</th>
      <td>53451</td>
    </tr>
    <tr>
      <th>5</th>
      <td>20663</td>
    </tr>
    <tr>
      <th>6</th>
      <td>13737</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3127</td>
    </tr>
  </tbody>
</table>



This is to apply the BMI_Category for the new synthetic data.

#### Feature Scaling

Since most of our variables are boolean, there are some that are numerical/integer, which may lead to biases when it comes to correlation.

To fix this issue, we can use feature scaling to put values between 0 and 1 to ensure no biases.


```python
from sklearn.preprocessing import MinMaxScaler
scale_cols = ['PhysHlth', 'MentHlth', 'GenHlth', 'Age', 'Education', 'Income', 'BMI', 'Healthy_Score']

scaler = MinMaxScaler()
X_train[scale_cols] = scaler.fit_transform(X_train[scale_cols])
X_test[scale_cols] = scaler.transform(X_test[scale_cols])

print(X_train[scale_cols].describe().loc[['min', 'max']])
```

         PhysHlth  MentHlth  GenHlth  Age  Education  Income  BMI  Healthy_Score
    min       0.0       0.0      0.0  0.0        0.0     0.0  0.0            0.0
    max       1.0       1.0      1.0  1.0        1.0     1.0  1.0            1.0
    

## Data Prediction Model

We have now done our cleaning and initial analysis, as well as made new features and scaled our data.

The next step will be to create a machine learning model that can predict if a patient has diabetes or not.

### Baseline Model

To begin, let us first use a simple ML model, first, Logistic Regression, and Random Forest Classifier. Both are simple classification models and will be used as a benchmark.



```python
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score, recall_score

lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train_res, y_train_res)
y_pred_lr = lr_model.predict(X_test)

rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_train_res, y_train_res)
y_pred_rf = rf_model.predict(X_test)

print("Logistic Regression Report:\n", classification_report(y_test, y_pred_lr))
print("Random Forest Report:\n", classification_report(y_test, y_pred_rf))
```

    /usr/local/lib/python3.12/dist-packages/sklearn/linear_model/_logistic.py:465: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      n_iter_i = _check_optimize_result(
    

    Logistic Regression Report:
                   precision    recall  f1-score   support
    
               0       0.89      0.51      0.65     43667
               1       0.17      0.63      0.27      7069
    
        accuracy                           0.52     50736
       macro avg       0.53      0.57      0.46     50736
    weighted avg       0.79      0.52      0.59     50736
    
    Random Forest Report:
                   precision    recall  f1-score   support
    
               0       0.86      0.99      0.92     43667
               1       0.36      0.02      0.04      7069
    
        accuracy                           0.86     50736
       macro avg       0.61      0.51      0.48     50736
    weighted avg       0.79      0.86      0.80     50736
    
    

As seen above, we can see that the Logistic Regression handles 63% of actual diabetes cases through the recall, however, it has a very low precision of 17%. When it predicts diabetes, it is only right 17% of the time.

This can be due to the use of SMOTE, creating a lot of records with diabetes but is very vulnerable to make mistakes.

As for the Random Forest, we have a somewhat opposite story. The recall shows 2%, indicating that it only catches 2% of diabetic patients. There is also an accuracy of 86%, but this is contradictory since most of our data are made up of non-diabetic patients, so guessing False for the target variable would naturally be mostly correct.

Despite our use of SMOTE, the model is overfitting to the non-diabetic class or is struggling with the use of synthetic noise.

### Main Model (XGBoost)

We will now be continuing with the use of XGBoost, which is an ML model that uses Gradient Boosting. It handles non-linearity well, it's good for analyzing bias and variance, and usually used in basic medical predictive tasks.


```python
if str(X_train_res['BMI_Category'].dtype) == 'category':
    X_train_res['BMI_Category'] = X_train_res['BMI_Category'].cat.codes
    X_test['BMI_Category'] = X_test['BMI_Category'].cat.codes
```


```python
from xgboost import XGBClassifier

ratio = float(y_train.value_counts()[0] / y_train.value_counts()[1])

xgb_model = XGBClassifier(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=5,
    scale_pos_weight=ratio,
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=42
)

xgb_model.fit(X_train_res, y_train_res)
y_pred_xgb = xgb_model.predict(X_test)

print("XGBoost Report:\n", classification_report(y_test, y_pred_xgb))
```

    /usr/local/lib/python3.12/dist-packages/xgboost/training.py:200: UserWarning: [06:52:32] WARNING: /__w/xgboost/xgboost/src/learner.cc:782: 
    Parameters: { "use_label_encoder" } are not used.
    
      bst.update(dtrain, iteration=i, fobj=obj)
    

    XGBoost Report:
                   precision    recall  f1-score   support
    
               0       0.87      0.81      0.84     43667
               1       0.17      0.24      0.19      7069
    
        accuracy                           0.73     50736
       macro avg       0.52      0.52      0.52     50736
    weighted avg       0.77      0.73      0.75     50736
    
    

The new results tell a similar story. It's is an improvement from the Random Forest, with a lower accuracy but can better predict cases of diabetes.

As a test, let us run a Randomized Search to tune the f1 score, making precision and recall more balanced.


```python
from sklearn.model_selection import RandomizedSearchCV

param_grid = {
    'max_depth': [3, 4, 5, 6],
    'learning_rate': [0.01, 0.05, 0.1],
    'n_estimators': [100, 200, 300],
    'colsample_bytree': [0.7, 0.8, 0.9],
    'subsample': [0.7, 0.8, 0.9]
}

random_search = RandomizedSearchCV(
    XGBClassifier(eval_metric='logloss', random_state=42),
    param_distributions=param_grid,
    n_iter=10,
    scoring='f1',
    cv=3,
    verbose=1,
    n_jobs=-1
)

random_search.fit(X_train_res, y_train_res)
best_xgb = random_search.best_estimator_

y_pred_opt = best_xgb.predict(X_test)
print("Optimized XGBoost Report:\n", classification_report(y_test, y_pred_opt))
```

    Fitting 3 folds for each of 10 candidates, totalling 30 fits
    Optimized XGBoost Report:
                   precision    recall  f1-score   support
    
               0       0.87      0.93      0.90     43667
               1       0.25      0.15      0.18      7069
    
        accuracy                           0.82     50736
       macro avg       0.56      0.54      0.54     50736
    weighted avg       0.78      0.82      0.80     50736
    
    

It seems that the Randomized Search did help a bit, but it's still not pretty good in predicting cases of diabetes.

To simply perform a sanity check, the next thing I'll do is use the original training and testing data.


```python
if str(X_train['BMI_Category'].dtype) == 'category':
    X_train['BMI_Category'] = X_train['BMI_Category'].cat.codes

```


```python
ratio = float(y_train.value_counts()[0] / y_train.value_counts()[1])

weighted_xgb = XGBClassifier(
    n_estimators=250,
    learning_rate=0.05,
    max_depth=6,
    scale_pos_weight=ratio,
    eval_metric='logloss',
    random_state=42
)

weighted_xgb.fit(X_train, y_train)
y_pred_weighted = weighted_xgb.predict(X_test)

print("Weighted XGBoost (No SMOTE) Report:\n", classification_report(y_test, y_pred_weighted))
```

    Weighted XGBoost (No SMOTE) Report:
                   precision    recall  f1-score   support
    
               0       0.95      0.71      0.82     43667
               1       0.31      0.79      0.44      7069
    
        accuracy                           0.72     50736
       macro avg       0.63      0.75      0.63     50736
    weighted avg       0.86      0.72      0.77     50736
    
    

We now have something that is better than the previous models. Turns out, not using SMOTE was more effective.

We achieved a recall of 79%, meaning it almost catches 80% of diabetic cases. Though, a 31% precision may be more desirable.


```python
importances = weighted_xgb.feature_importances_
feature_names = X.columns
feat_importances = pd.Series(importances, index=feature_names)

plt.figure(figsize=(10,6))
feat_importances.nlargest(10).sort_values().plot(kind='barh', color='teal')
plt.title('Top 10 Predictors for Diabetes (Weighted XGBoost)')
plt.xlabel('F-Score (Importance)')
plt.show()
```


    
![png](diabetes_notebook_files/diabetes_notebook_67_0.png)
    


Upon checking, we see that HighBP, GenHlth, and HighChol are the leading three fields, with Age and BMI following use.
- HighBP & HighChol: These usually dominate. They are the strongest physiological indicators.
- GenHlth: Patient self-perception is a surprisingly powerful predictor.
- BMI: Important but not a smoking gun factor.
- Age: Diabetes risk scales heavily with a patients' age.

### Model Optimization

Now that we have completed creating our model, we can try one last optimization before we can finish.

In this setup, we will run an optuna optimizer to determine the best parameters for the XGBoost model, but only running through 20 trials as to preserve memory.


```python
import optuna
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer, recall_score, precision_score

def objective(trial):
    param = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'max_depth': trial.suggest_int('max_depth', 3, 9),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'gamma': trial.suggest_float('gamma', 0, 5),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'scale_pos_weight': ratio
    }

    model = XGBClassifier(**param, eval_metric='logloss', random_state=42)

    score = cross_val_score(model, X_train, y_train, cv=3, scoring='f1').mean()
    return score

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=20)

print("Best Parameters:", study.best_params)
```

    [I 2026-02-24 07:12:16,114] A new study created in memory with name: no-name-f5ee493e-2df5-473b-af84-602bfab8b4ce
    [I 2026-02-24 07:12:54,281] Trial 0 finished with value: 0.4482610698088396 and parameters: {'n_estimators': 119, 'max_depth': 9, 'learning_rate': 0.015506122541211887, 'subsample': 0.6006090987588006, 'colsample_bytree': 0.7535162149767414, 'gamma': 3.7208337726112384, 'min_child_weight': 2}. Best is trial 0 with value: 0.4482610698088396.
    [I 2026-02-24 07:13:05,181] Trial 1 finished with value: 0.44420544385812155 and parameters: {'n_estimators': 141, 'max_depth': 7, 'learning_rate': 0.030900447584382342, 'subsample': 0.9156942437133933, 'colsample_bytree': 0.8260988784707297, 'gamma': 3.72436024687845, 'min_child_weight': 2}. Best is trial 0 with value: 0.4482610698088396.
    [I 2026-02-24 07:13:27,810] Trial 2 finished with value: 0.4444168516190891 and parameters: {'n_estimators': 429, 'max_depth': 5, 'learning_rate': 0.03223774881375672, 'subsample': 0.622526277027248, 'colsample_bytree': 0.8222628948144651, 'gamma': 0.4103053576464083, 'min_child_weight': 9}. Best is trial 0 with value: 0.4482610698088396.
    [I 2026-02-24 07:13:49,446] Trial 3 finished with value: 0.44354460069325347 and parameters: {'n_estimators': 456, 'max_depth': 4, 'learning_rate': 0.03681570748225612, 'subsample': 0.9294732572061217, 'colsample_bytree': 0.8999635917106814, 'gamma': 2.8198958846572113, 'min_child_weight': 2}. Best is trial 0 with value: 0.4482610698088396.
    [I 2026-02-24 07:14:09,604] Trial 4 finished with value: 0.4424285739027336 and parameters: {'n_estimators': 451, 'max_depth': 4, 'learning_rate': 0.012525553884515995, 'subsample': 0.7501947324079262, 'colsample_bytree': 0.6609749759843672, 'gamma': 2.576189981535668, 'min_child_weight': 3}. Best is trial 0 with value: 0.4482610698088396.
    [I 2026-02-24 07:14:22,957] Trial 5 finished with value: 0.44324129738121537 and parameters: {'n_estimators': 295, 'max_depth': 4, 'learning_rate': 0.03528564167433342, 'subsample': 0.9060718604406988, 'colsample_bytree': 0.7426270456002846, 'gamma': 0.604608979558367, 'min_child_weight': 9}. Best is trial 0 with value: 0.4482610698088396.
    [I 2026-02-24 07:14:37,332] Trial 6 finished with value: 0.4443611470675304 and parameters: {'n_estimators': 200, 'max_depth': 6, 'learning_rate': 0.03806261982510978, 'subsample': 0.7040551273852045, 'colsample_bytree': 0.6873621718906164, 'gamma': 0.2303726956647867, 'min_child_weight': 9}. Best is trial 0 with value: 0.4482610698088396.
    [I 2026-02-24 07:14:47,274] Trial 7 finished with value: 0.44363471138227334 and parameters: {'n_estimators': 180, 'max_depth': 5, 'learning_rate': 0.04812529054393407, 'subsample': 0.9203401341722993, 'colsample_bytree': 0.7007239985767231, 'gamma': 4.098464121333731, 'min_child_weight': 10}. Best is trial 0 with value: 0.4482610698088396.
    [I 2026-02-24 07:15:26,371] Trial 8 finished with value: 0.4471480911608922 and parameters: {'n_estimators': 424, 'max_depth': 8, 'learning_rate': 0.01849108278252561, 'subsample': 0.9633946256277617, 'colsample_bytree': 0.8087544967284392, 'gamma': 3.6084317196179736, 'min_child_weight': 9}. Best is trial 0 with value: 0.4482610698088396.
    [I 2026-02-24 07:15:59,410] Trial 9 finished with value: 0.439990291892206 and parameters: {'n_estimators': 418, 'max_depth': 9, 'learning_rate': 0.07574318163707383, 'subsample': 0.6673410650277932, 'colsample_bytree': 0.8793103783052016, 'gamma': 2.240763436604451, 'min_child_weight': 9}. Best is trial 0 with value: 0.4482610698088396.
    [I 2026-02-24 07:16:10,872] Trial 10 finished with value: 0.44448438726683087 and parameters: {'n_estimators': 103, 'max_depth': 9, 'learning_rate': 0.010195988707318658, 'subsample': 0.8164056749270182, 'colsample_bytree': 0.9934705384681022, 'gamma': 4.7528454438960805, 'min_child_weight': 5}. Best is trial 0 with value: 0.4482610698088396.
    [I 2026-02-24 07:16:36,899] Trial 11 finished with value: 0.44745394086387263 and parameters: {'n_estimators': 335, 'max_depth': 8, 'learning_rate': 0.01837707968294674, 'subsample': 0.9941236824372853, 'colsample_bytree': 0.607745453403723, 'gamma': 3.466855830215876, 'min_child_weight': 6}. Best is trial 0 with value: 0.4482610698088396.
    [I 2026-02-24 07:16:59,889] Trial 12 finished with value: 0.44785005762654895 and parameters: {'n_estimators': 315, 'max_depth': 8, 'learning_rate': 0.01909242567993383, 'subsample': 0.8178853420337244, 'colsample_bytree': 0.6047195386430311, 'gamma': 3.0497984869977, 'min_child_weight': 6}. Best is trial 0 with value: 0.4482610698088396.
    [I 2026-02-24 07:17:21,123] Trial 13 finished with value: 0.4477611148856282 and parameters: {'n_estimators': 266, 'max_depth': 8, 'learning_rate': 0.01809644817983324, 'subsample': 0.8223004309928124, 'colsample_bytree': 0.6011808954276215, 'gamma': 1.6897525009176435, 'min_child_weight': 6}. Best is trial 0 with value: 0.4482610698088396.
    [I 2026-02-24 07:17:43,802] Trial 14 finished with value: 0.4459415981072159 and parameters: {'n_estimators': 352, 'max_depth': 7, 'learning_rate': 0.022921521995272903, 'subsample': 0.8448086867291381, 'colsample_bytree': 0.7487246842045214, 'gamma': 4.889955332268813, 'min_child_weight': 4}. Best is trial 0 with value: 0.4482610698088396.
    [I 2026-02-24 07:18:06,135] Trial 15 finished with value: 0.4483254210872678 and parameters: {'n_estimators': 228, 'max_depth': 9, 'learning_rate': 0.01395563069181798, 'subsample': 0.7629702963858808, 'colsample_bytree': 0.7567284686709179, 'gamma': 1.4740862967937154, 'min_child_weight': 1}. Best is trial 15 with value: 0.4483254210872678.
    [I 2026-02-24 07:18:26,411] Trial 16 finished with value: 0.44808912353608993 and parameters: {'n_estimators': 225, 'max_depth': 9, 'learning_rate': 0.0130199087488465, 'subsample': 0.7486646338680158, 'colsample_bytree': 0.7549315457742009, 'gamma': 1.3236130068772185, 'min_child_weight': 1}. Best is trial 15 with value: 0.4483254210872678.
    [I 2026-02-24 07:18:35,749] Trial 17 finished with value: 0.44312395374842256 and parameters: {'n_estimators': 101, 'max_depth': 7, 'learning_rate': 0.013662617280615715, 'subsample': 0.606905449927654, 'colsample_bytree': 0.869208929549522, 'gamma': 1.1518778519955175, 'min_child_weight': 1}. Best is trial 15 with value: 0.4483254210872678.
    [I 2026-02-24 07:18:46,331] Trial 18 finished with value: 0.4412394631429229 and parameters: {'n_estimators': 244, 'max_depth': 3, 'learning_rate': 0.024164619357350384, 'subsample': 0.6759212622114799, 'colsample_bytree': 0.7642817127324686, 'gamma': 2.091296339728154, 'min_child_weight': 3}. Best is trial 15 with value: 0.4483254210872678.
    [I 2026-02-24 07:19:02,075] Trial 19 finished with value: 0.4466067724365524 and parameters: {'n_estimators': 162, 'max_depth': 9, 'learning_rate': 0.010372453383869837, 'subsample': 0.7521214693001973, 'colsample_bytree': 0.929314206356996, 'gamma': 4.230549962602481, 'min_child_weight': 4}. Best is trial 15 with value: 0.4483254210872678.
    

    Best Parameters: {'n_estimators': 228, 'max_depth': 9, 'learning_rate': 0.01395563069181798, 'subsample': 0.7629702963858808, 'colsample_bytree': 0.7567284686709179, 'gamma': 1.4740862967937154, 'min_child_weight': 1}
    

Next, let us apply the determined best parameters.


```python
ratio = float(y_train.value_counts()[0] / y_train.value_counts()[1])

weighted_xgb = XGBClassifier(
    n_estimators=228,
    learning_rate=0.01395563069181798,
    max_depth=9,
    subsample=0.7629702963858808,
    colsample_bytree=0.7567284686709179,
    gamma=1.4740862967937154,
    min_child_weight=1,
    scale_pos_weight=ratio,
    eval_metric='logloss',
    random_state=42
)

weighted_xgb.fit(X_train, y_train)
y_pred_weighted = weighted_xgb.predict(X_test)

print("Weighted XGBoost (No SMOTE) Report:\n", classification_report(y_test, y_pred_weighted))
```

    Weighted XGBoost (No SMOTE) Report:
                   precision    recall  f1-score   support
    
               0       0.95      0.73      0.82     43667
               1       0.31      0.77      0.45      7069
    
        accuracy                           0.73     50736
       macro avg       0.63      0.75      0.63     50736
    weighted avg       0.86      0.73      0.77     50736
    
    

We have a slight improvement in terms of accuracy and f1-score for identifying diabetic cases, but a slight dip and improvement in the recall of diabetic and non-diabetic classification.

We can conclude the model evaluation here.

## Interpretation

The modeling phase revealed that a cost-sensitive learning approach is significantly better than using synthetic data for this dataset.

The Weighted XGBoost set a relatively high baseline, with a Recall of 0.79, meaning it successfully identifies 79% of individuals with diabetes in a test set, but is only precise for 31% of those cases.

Based on XGBoost’s feature importance, the following variables were the primary drivers of the classification:
- High Blood Pressure & High Cholesterol: These remained the strongest physiological indicators.
- General Health (GenHlth): Patient self-perception was a top-tier predictor.
- BMI & Age: These were secondary to the direct metabolic markers above.
- Healthy_Score: This indicates that the combination of lifestyle factors (smoking, diet, alcohol) provides predictive value beyond single variables.


## Recommendation

For future iterations on this dataset, use scale_pos_weight or class weights rather than SMOTE to maintain the integrity of patient profiles.

To ensure that the output considers real-wold risks, future projects may apply Platt Scaling or Isotonic Regression to the final XGBoost model.
