
# Project 2: Supervised Learning
### Building a Student Intervention System

## 2. Exploring the Data

Let's go ahead and read in the student dataset first.

_To execute a code cell, click inside it and press **Shift+Enter**._


```python
# Import libraries
import numpy as np
import pandas as pd
```


```python
# Read student data
student_data = pd.read_csv("student-data.csv")
print "Student data read successfully!"
# Note: The last column 'passed' is the target/label, all other are feature columns
```

    Student data read successfully!
    


```python
# TODO: Compute desired values - replace each '?' with an appropriate expression/function call
n_students = np.shape(student_data)[0]
n_features = np.shape(student_data)[1] - 1 # Subtract target column
n_passed = np.shape(student_data[student_data['passed']=='yes'])[0]
n_failed = np.shape(student_data[student_data['passed']=='no'])[0]
grad_rate = float(n_passed) / float(n_students)*100
print "Total number of students: {}".format(n_students)
print "Number of students who passed: {}".format(n_passed)
print "Number of students who failed: {}".format(n_failed)
print "Number of features: {}".format(n_features)
print "Graduation rate of the class: {:.2f}%".format(grad_rate)
```

    Total number of students: 395
    Number of students who passed: 265
    Number of students who failed: 130
    Number of features: 30
    Graduation rate of the class: 67.09%
    

## 3. Preparing the Data
In this section, we will prepare the data for modeling, training and testing.

### Identify feature and target columns
It is often the case that the data you obtain contains non-numeric features. This can be a problem, as most machine learning algorithms expect numeric data to perform computations with.

Let's first separate our data into feature and target columns, and see if any features are non-numeric.<br/>
**Note**: For this dataset, the last column (`'passed'`) is the target or label we are trying to predict.


```python
# Extract feature (X) and target (y) columns
feature_cols = list(student_data.columns[:-1])  # all columns but last are features
target_col = student_data.columns[-1]  # last column is the target/label
print "Feature column(s):-\n{}".format(feature_cols)
print "Target column: {}".format(target_col)

X_all = student_data[feature_cols]  # feature values for all students
y_all = student_data[target_col]  # corresponding targets/labels
print "\nFeature values:-"
print X_all.head()  # print the first 5 rows
```

    Feature column(s):-
    ['school', 'sex', 'age', 'address', 'famsize', 'Pstatus', 'Medu', 'Fedu', 'Mjob', 'Fjob', 'reason', 'guardian', 'traveltime', 'studytime', 'failures', 'schoolsup', 'famsup', 'paid', 'activities', 'nursery', 'higher', 'internet', 'romantic', 'famrel', 'freetime', 'goout', 'Dalc', 'Walc', 'health', 'absences']
    Target column: passed
    
    Feature values:-
      school sex  age address famsize Pstatus  Medu  Fedu     Mjob      Fjob  \
    0     GP   F   18       U     GT3       A     4     4  at_home   teacher   
    1     GP   F   17       U     GT3       T     1     1  at_home     other   
    2     GP   F   15       U     LE3       T     1     1  at_home     other   
    3     GP   F   15       U     GT3       T     4     2   health  services   
    4     GP   F   16       U     GT3       T     3     3    other     other   
    
        ...    higher internet  romantic  famrel  freetime goout Dalc Walc health  \
    0   ...       yes       no        no       4         3     4    1    1      3   
    1   ...       yes      yes        no       5         3     3    1    1      3   
    2   ...       yes      yes        no       4         3     2    2    3      3   
    3   ...       yes      yes       yes       3         2     2    1    1      5   
    4   ...       yes       no        no       4         3     2    1    2      5   
    
      absences  
    0        6  
    1        4  
    2       10  
    3        2  
    4        4  
    
    [5 rows x 30 columns]
    

### Preprocess feature columns

As you can see, there are several non-numeric columns that need to be converted! Many of them are simply `yes`/`no`, e.g. `internet`. These can be reasonably converted into `1`/`0` (binary) values.

Other columns, like `Mjob` and `Fjob`, have more than two values, and are known as _categorical variables_. The recommended way to handle such a column is to create as many columns as possible values (e.g. `Fjob_teacher`, `Fjob_other`, `Fjob_services`, etc.), and assign a `1` to one of them and `0` to all others.

These generated columns are sometimes called _dummy variables_, and we will use the [`pandas.get_dummies()`](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.get_dummies.html?highlight=get_dummies#pandas.get_dummies) function to perform this transformation.


```python
# Preprocess feature columns
def preprocess_features(X):
    outX = pd.DataFrame(index=X.index)  # output dataframe, initially empty

    # Check each column
    for col, col_data in X.iteritems():
        # If data type is non-numeric, try to replace all yes/no values with 1/0
        if col_data.dtype == object:
            col_data = col_data.replace(['yes', 'no'], [1, 0])
        # Note: This should change the data type for yes/no columns to int

        # If still non-numeric, convert to one or more dummy variables
        if col_data.dtype == object:
            col_data = pd.get_dummies(col_data, prefix=col)  # e.g. 'school' => 'school_GP', 'school_MS'

        outX = outX.join(col_data)  # collect column(s) in output dataframe

    return outX

X_all = preprocess_features(X_all)
print "Processed feature columns ({}):-\n{}".format(len(X_all.columns), list(X_all.columns))
```

    Processed feature columns (48):-
    ['school_GP', 'school_MS', 'sex_F', 'sex_M', 'age', 'address_R', 'address_U', 'famsize_GT3', 'famsize_LE3', 'Pstatus_A', 'Pstatus_T', 'Medu', 'Fedu', 'Mjob_at_home', 'Mjob_health', 'Mjob_other', 'Mjob_services', 'Mjob_teacher', 'Fjob_at_home', 'Fjob_health', 'Fjob_other', 'Fjob_services', 'Fjob_teacher', 'reason_course', 'reason_home', 'reason_other', 'reason_reputation', 'guardian_father', 'guardian_mother', 'guardian_other', 'traveltime', 'studytime', 'failures', 'schoolsup', 'famsup', 'paid', 'activities', 'nursery', 'higher', 'internet', 'romantic', 'famrel', 'freetime', 'goout', 'Dalc', 'Walc', 'health', 'absences']
    

### Split data into training and test sets

So far, we have converted all _categorical_ features into numeric values. In this next step, we split the data (both features and corresponding labels) into training and test sets.


```python
import sklearn.cross_validation as cv


# First, decide how many training vs test samples you want
num_all = student_data.shape[0]  # same as len(student_data)
num_train = 300  # about 75% of the data
num_test = num_all - num_train
num_ratio = float(num_train) / float(num_all)

# TODO: Then, select features (X) and corresponding labels (y) for the training and test sets
# Note: Shuffle the data or randomly select samples to avoid any bias due to ordering in the dataset

X_train, X_test, y_train, y_test = cv.train_test_split(X_all, y_all, test_size=(1 - num_ratio), random_state=1234)



print "Training set: {} samples".format(X_train.shape[0])
print "Test set: {} samples".format(X_test.shape[0])
# Note: If you need a validation set, extract it from within training data
```

    Training set: 300 samples
    Test set: 95 samples
    

## 4. Training and Evaluating Models
Choose 3 supervised learning models that are available in scikit-learn, and appropriate for this problem. For each model:

- What are the general applications of this model? What are its strengths and weaknesses?
- Given what you know about the data so far, why did you choose this model to apply?
- Fit this model to the training data, try to predict labels (for both training and test sets), and measure the F<sub>1</sub> score. Repeat this process with different training set sizes (100, 200, 300), keeping test set constant.

Produce a table showing training time, prediction time, F<sub>1</sub> score on training set and F<sub>1</sub> score on test set, for each training set size.

Note: You need to produce 3 such tables - one for each model.


```python
# Train a model
import time

#timetotrain = []
def train_classifier(clf, X_train, y_train):
    print "Training {}...".format(clf.__class__.__name__)
    start = time.time()
    clf.fit(X_train, y_train)
    end = time.time()
    print "Done!\nTraining time (secs): {:.3f}".format(end - start)
    #timetotrain.append(end - start)

# TODO: Choose a model, import it and instantiate an object
from sklearn.linear_model import LogisticRegression 
#from sklearn.neural_network import MLPClassifier
#from sklearn.ensemble import RandomForestClassifier
clf = LogisticRegression()

# Fit model to training data
train_classifier(clf, X_train, y_train)

# note: using entire training set here
#print clf  # you can inspect the learned model by printing it
```

    Training LogisticRegression...
    Done!
    Training time (secs): 0.005
    


```python
# Predict on training set and compute F1 score
from sklearn.metrics import f1_score

def predict_labels(clf, features, target):
    print "Predicting labels using {}...".format(clf.__class__.__name__)
    start = time.time()
    y_pred = clf.predict(features)
    end = time.time()
    print "Done!\nPrediction time (secs): {:.3f}".format(end - start)
    return f1_score(target.values, y_pred, pos_label='yes')

train_f1_score = predict_labels(clf, X_train, y_train)
print "F1 score for training set: {}".format(train_f1_score)
```

    Predicting labels using LogisticRegression...
    Done!
    Prediction time (secs): 0.001
    F1 score for training set: 0.831050228311
    


```python
# Predict on test data
print "F1 score for test set: {}".format(predict_labels(clf, X_test, y_test))
```

    Predicting labels using LogisticRegression...
    Done!
    Prediction time (secs): 0.000
    F1 score for test set: 0.8
    


```python
# Train and predict using different training set sizes

def train_predict(clf, X_train, y_train, X_test, y_test):
    print "------------------------------------------"
    print "Training set size: {}".format(len(X_train))
    train_classifier(clf, X_train, y_train)
    print "F1 score for training set: {}".format(predict_labels(clf, X_train, y_train))
    print "F1 score for test set: {}".format(predict_labels(clf, X_test, y_test))


# TODO: Run the helper function above for desired subsets of training data
# Note: Keep the test set constant

# Sample slices the DF/Series and also randomizes
X_train_100 = pd.DataFrame.sample(X_train, n = 100, random_state = 1234)
y_train_100 =    pd.Series.sample(y_train, n = 100, random_state = 1234)
X_train_200 = pd.DataFrame.sample(X_train, n = 200, random_state = 1234)
y_train_200 =    pd.Series.sample(y_train, n = 200, random_state = 1234)
X_train_300 = pd.DataFrame.sample(X_train, n = 300, random_state = 1234)
y_train_300 =    pd.Series.sample(y_train, n = 300, random_state = 1234)


train_predict(clf, X_train_100, y_train_100, X_test, y_test);
train_predict(clf, X_train_200, y_train_200, X_test, y_test);
train_predict(clf, X_train_300, y_train_300, X_test, y_test);
```

    ------------------------------------------
    Training set size: 100
    Training LogisticRegression...
    Done!
    Training time (secs): 0.003
    Predicting labels using LogisticRegression...
    Done!
    Prediction time (secs): 0.000
    F1 score for training set: 0.906832298137
    Predicting labels using LogisticRegression...
    Done!
    Prediction time (secs): 0.000
    F1 score for test set: 0.759124087591
    ------------------------------------------
    Training set size: 200
    Training LogisticRegression...
    Done!
    Training time (secs): 0.003
    Predicting labels using LogisticRegression...
    Done!
    Prediction time (secs): 0.000
    F1 score for training set: 0.865979381443
    Predicting labels using LogisticRegression...
    Done!
    Prediction time (secs): 0.001
    F1 score for test set: 0.788321167883
    ------------------------------------------
    Training set size: 300
    Training LogisticRegression...
    Done!
    Training time (secs): 0.005
    Predicting labels using LogisticRegression...
    Done!
    Prediction time (secs): 0.000
    F1 score for training set: 0.831050228311
    Predicting labels using LogisticRegression...
    Done!
    Prediction time (secs): 0.001
    F1 score for test set: 0.8
    


```python
# While playing around with the training set size, I tried incredibly low numbers to see how they performed. 
# That's when I stumbled on the fact that a training size of 11 actually performed better than using a size of 300. 
# I'm not sure of the exact reasons behind this, but would be interested to learn why.

X_train_11 = pd.DataFrame.sample(X_train, n = 11, random_state = 1234)
y_train_11 =    pd.Series.sample(y_train, n = 11, random_state = 1234)

train_predict(clf, X_train_11, y_train_11, X_test, y_test)
```

    ------------------------------------------
    Training set size: 11
    Training LogisticRegression...
    Done!
    Training time (secs): 0.001
    Predicting labels using LogisticRegression...
    Done!
    Prediction time (secs): 0.000
    F1 score for training set: 1.0
    Predicting labels using LogisticRegression...
    Done!
    Prediction time (secs): 0.000
    F1 score for test set: 0.805031446541
    


```python
# TODO: Train and predict using two other models

# Train and predict a basic decision tree classifier
from sklearn import tree
clf = tree.DecisionTreeClassifier()
train_predict(clf, X_train_100, y_train_100, X_test, y_test)
train_predict(clf, X_train_200, y_train_200, X_test, y_test)
train_predict(clf, X_train_300, y_train_300, X_test, y_test)

# Train and predict a gradient boosting classifier
from sklearn.ensemble import GradientBoostingClassifier
clf = GradientBoostingClassifier(n_estimators=100, learning_rate=.1, max_depth=3, random_state=1234).fit(X_train, y_train)
train_predict(clf, X_train_100, y_train_100, X_test, y_test)
train_predict(clf, X_train_200, y_train_200, X_test, y_test)
train_predict(clf, X_train_300, y_train_300, X_test, y_test)


```

    ------------------------------------------
    Training set size: 100
    Training DecisionTreeClassifier...
    Done!
    Training time (secs): 0.010
    Predicting labels using DecisionTreeClassifier...
    Done!
    Prediction time (secs): 0.001
    F1 score for training set: 1.0
    Predicting labels using DecisionTreeClassifier...
    Done!
    Prediction time (secs): 0.000
    F1 score for test set: 0.744186046512
    ------------------------------------------
    Training set size: 200
    Training DecisionTreeClassifier...
    Done!
    Training time (secs): 0.002
    Predicting labels using DecisionTreeClassifier...
    Done!
    Prediction time (secs): 0.000
    F1 score for training set: 1.0
    Predicting labels using DecisionTreeClassifier...
    Done!
    Prediction time (secs): 0.000
    F1 score for test set: 0.709677419355
    ------------------------------------------
    Training set size: 300
    Training DecisionTreeClassifier...
    Done!
    Training time (secs): 0.003
    Predicting labels using DecisionTreeClassifier...
    Done!
    Prediction time (secs): 0.001
    F1 score for training set: 1.0
    Predicting labels using DecisionTreeClassifier...
    Done!
    Prediction time (secs): 0.000
    F1 score for test set: 0.650406504065
    ------------------------------------------
    Training set size: 100
    Training GradientBoostingClassifier...
    Done!
    Training time (secs): 0.078
    Predicting labels using GradientBoostingClassifier...
    Done!
    Prediction time (secs): 0.003
    F1 score for training set: 1.0
    Predicting labels using GradientBoostingClassifier...
    Done!
    Prediction time (secs): 0.001
    F1 score for test set: 0.785185185185
    ------------------------------------------
    Training set size: 200
    Training GradientBoostingClassifier...
    Done!
    Training time (secs): 0.109
    Predicting labels using GradientBoostingClassifier...
    Done!
    Prediction time (secs): 0.001
    F1 score for training set: 0.992805755396
    Predicting labels using GradientBoostingClassifier...
    Done!
    Prediction time (secs): 0.001
    F1 score for test set: 0.761194029851
    ------------------------------------------
    Training set size: 300
    Training GradientBoostingClassifier...
    Done!
    Training time (secs): 0.136
    Predicting labels using GradientBoostingClassifier...
    Done!
    Prediction time (secs): 0.001
    F1 score for training set: 0.97572815534
    Predicting labels using GradientBoostingClassifier...
    Done!
    Prediction time (secs): 0.000
    F1 score for test set: 0.820143884892
    


```python
# Create the table / DataFrame - Logistic Regression

columns = ['Training set size:','100','200','300']
data = np.array([['Training time (secs)',0.003,0.003,0.005], ['Prediction time (secs)',0.000,0.001,0.001],['F1 score for training set',0.90683,0.86598,0.83105],['F1 score for test set',0.75912,0.788321,0.8]])

LogRegTable = pd.DataFrame(data, columns = columns)

# Create the table / DataFrame - Decision Tree

columns = ['Training set size:','100','200','300']
data2 = np.array([['Training time (secs)',0.010,0.002,0.003], ['Prediction time (secs)',0.000,0.000,0.000],['F1 score for training set',1.0,1.0,1.0],['F1 score for test set',0.74419,0.70967,0.650407]])

DecTreeTable = pd.DataFrame(data2, columns = columns)

# Create the table / DataFrame - Gradient Boosting

columns = ['Training set size:','100','200','300']
data3 = np.array([['Training time (secs)',0.078,0.109,0.136], ['Prediction time (secs)',0.001,0.001,0.000],['F1 score for training set',1.0,0.99281,0.975728],['F1 score for test set',0.78519,0.761194,0.821439]])

GradBoostTable = pd.DataFrame(data3, columns = columns)
```


```python
LogRegTable
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Training set size:</th>
      <th>100</th>
      <th>200</th>
      <th>300</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Training time (secs)</td>
      <td>0.003</td>
      <td>0.003</td>
      <td>0.005</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Prediction time (secs)</td>
      <td>0.0</td>
      <td>0.001</td>
      <td>0.001</td>
    </tr>
    <tr>
      <th>2</th>
      <td>F1 score for training set</td>
      <td>0.90683</td>
      <td>0.86598</td>
      <td>0.83105</td>
    </tr>
    <tr>
      <th>3</th>
      <td>F1 score for test set</td>
      <td>0.75912</td>
      <td>0.788321</td>
      <td>0.8</td>
    </tr>
  </tbody>
</table>
</div>




```python
DecTreeTable
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Training set size:</th>
      <th>100</th>
      <th>200</th>
      <th>300</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Training time (secs)</td>
      <td>0.01</td>
      <td>0.002</td>
      <td>0.003</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Prediction time (secs)</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>F1 score for training set</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>F1 score for test set</td>
      <td>0.74419</td>
      <td>0.70967</td>
      <td>0.650407</td>
    </tr>
  </tbody>
</table>
</div>




```python
GradBoostTable
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Training set size:</th>
      <th>100</th>
      <th>200</th>
      <th>300</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Training time (secs)</td>
      <td>0.078</td>
      <td>0.109</td>
      <td>0.136</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Prediction time (secs)</td>
      <td>0.001</td>
      <td>0.001</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>F1 score for training set</td>
      <td>1.0</td>
      <td>0.99281</td>
      <td>0.975728</td>
    </tr>
    <tr>
      <th>3</th>
      <td>F1 score for test set</td>
      <td>0.78519</td>
      <td>0.761194</td>
      <td>0.821439</td>
    </tr>
  </tbody>
</table>
</div>




```python
# TODO: Fine-tune your model and report the best F1 score
from sklearn import grid_search
from sklearn.metrics import f1_score
from sklearn.metrics import make_scorer
f1_scorer = make_scorer(f1_score, pos_label="yes")


# Set the parameters to search, Logistic Regression is relatively simple, not many parameters
myparameters = {'C': [0.0001, 0.001, 0.01,0.05, 0.1,0.5, 1,5, 10, 100, 500,1000, 10000] }
clf = grid_search.GridSearchCV(LogisticRegression(penalty='l2'), scoring = f1_scorer, param_grid = myparameters)

train_predict(clf, X_train_300, y_train_300, X_test, y_test)

```

    ------------------------------------------
    Training set size: 300
    Training GridSearchCV...
    Done!
    Training time (secs): 0.356
    Predicting labels using GridSearchCV...
    Done!
    Prediction time (secs): 0.000
    F1 score for training set: 0.802395209581
    Predicting labels using GridSearchCV...
    Done!
    Prediction time (secs): 0.001
    F1 score for test set: 0.805031446541
    

### - What is the model's final F<sub>1</sub> score?

#### Answer:

  After tuning for possible parameter values, I am only able to obtain an 80.5% F<sub>1</sub> score. Which is just slightly higher than what the model was able to get before the grid search, at 80%.
