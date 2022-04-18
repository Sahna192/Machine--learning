# Machine--learning
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


np.random.seed(0)
n = 15
x = np.linspace(0,10,n) + np.random.randn(n)/5
y = np.sin(x)+x/6 + np.random.randn(n)/10


X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=0)
def part1_scatter():
    import matplotlib.pyplot as plt
    get_ipython().magic('matplotlib notebook')
    plt.figure()
    plt.scatter(X_train, y_train, label='training data')
    plt.scatter(X_test, y_test, label='test data')
    plt.legend(loc=4);
    def answer_one():
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import PolynomialFeatures
    
    result = np.zeros((4,100))
    
    # Your code here
    for i, degree in enumerate([1,3,6,9]):
        poly = PolynomialFeatures(degree=degree)
        X_poly = poly.fit_transform(X_train.reshape(11,1))
        linreg = LinearRegression().fit(X_poly, y_train)
        y = linreg.predict(poly.fit_transform(np.linspace(0,10,100).reshape(100,1)));
        result[i,:] = y
    return result # Return your answer
    def plot_one(degree_predictions):
    import matplotlib.pyplot as plt
    get_ipython().magic('matplotlib notebook')
    plt.figure(figsize=(10,5))
    plt.plot(X_train, y_train, 'o', label='training data', markersize=10)
    plt.plot(X_test, y_test, 'o', label='test data', markersize=10)
    for i,degree in enumerate([1,3,6,9]):
        plt.plot(np.linspace(0,10,100), degree_predictions[i], alpha=0.8, lw=2, label='degree={}'.format(degree))
    plt.ylim(-1,2.5)
    plt.legend(loc=4)
def answer_three():
    
    r2_scores = answer_two()
    df = pd.DataFrame({'training_score':r2_scores[0], 'test_score':r2_scores[1]})
    df['diff'] = df['training_score'] - df['test_score']
    
    df = df.sort(['diff'])
    good_gen = df.index[0]
    
    df = df.sort(['diff'], ascending = False)
    overfitting = df.index[0]
    
    df = df.sort(['training_score'])
    underfitting = df.index[0]
    
    return (underfitting,overfitting,good_gen)
def answer_four():
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.linear_model import Lasso, LinearRegression
    from sklearn.metrics.regression import r2_score

    # Create Polinomial Features
    poly = PolynomialFeatures(degree=12)
    
    # Shape Polinomial Features
    X_train_poly = poly.fit_transform(X_train.reshape(11,1))
    X_test_poly = poly.fit_transform(X_test.reshape(4,1))
    
    # Linear Regression
    linreg = LinearRegression().fit(X_train_poly, y_train)
    lin_r2_test = linreg.score(X_test_poly, y_test)

    # Lasso Regression
    linlasso = Lasso(alpha=0.01, max_iter = 10000).fit(X_train_poly, y_train)
    las_r2_test = linlasso.score(X_test_poly, y_test)
    
    return (lin_r2_test, las_r2_test) # Your answer here
    import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

mush_df = pd.read_csv('mushrooms.csv')
mush_df2 = pd.get_dummies(mush_df)

X_mush = mush_df2.iloc[:,2:]
y_mush = mush_df2.iloc[:,1]

# use the variables X_train2, y_train2 for Question 5
X_train2, X_test2, y_train2, y_test2 = train_test_split(X_mush, y_mush, random_state=0)

# For performance reasons in Questions 6 and 7, we will create a smaller version of the
# entire mushroom dataset for use in those questions.  For simplicity we'll just re-use
# the 25% test split created above as the representative subset.
#
# Use the variables X_subset, y_subset for Questions 6 and 7.
X_subset = X_test2
y_subset = y_test2
