import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from bokeh.plotting import figure, output_file, show
from bokeh.io import output_file, show, output_notebook
from bokeh.layouts import row, column

def boxplot(feature):
    """
    function to plot a boxplot
    Parameters
    -------------
    feature: 
        a pandas data frame object on which you want to plot box plot of
    -------------
    Return
        plot object - contains a box plot
    """
    return sns.boxplot(feature)

def distributionPlot(feature):
    """
    function to plot a distribution plot
    Parameters
    -------------
    feature: 
        a pandas data frame object on which you want to plot box plot of
    -------------
    Return
        plot object - contains a distribution plot plot
    """
    return sns.distplot(feature)

def ReplaceWithMedian(feature,df):
    """
    function to replace all missing values with the median off the distribution
    Parameters
    -------------
    feature: 
        Feature on which you want to perform this operation
    df: 
        Data frame which contains all the data
    -------------
    Return
    DataFrame:
        the dataframe where all the missing values of the varible feature are filled with median values
    """
    return df[feature].fillna(df[feature].median(),inplace=True)

def data_transform(df, feature_list):
    """
    This function will apply square root followed by log function to transform the data
    -------------
    Parameters 
    df: 
        The dataframe to which you want to apply this transforms on
    feature_list:
        List of features to which this particular transofrms should be applied to
    -------------
    Return
    DataFrame:
        Transformed data frame
    """
    df[feature_list] = np.sqrt(df[feature_list])
    df[feature_list] = np.log(df[feature_list])
    return df

def remove_outliers(feature,y1):
    """
    function to import data from a json file
    Parameters
    -------------
    filename: 
        name of the file  
    Return
    -------------
    Series or DataFrame:
        The contents of the file are returned in the Series or DataFrame format
    """
    iqr_pm25 = y1[feature].quantile(0.75) - y1[feature].quantile(0.25)
    low = y1[feature].quantile(0.25) - 1.5*iqr_pm25
    upp =  y1[feature].quantile(0.75) + 1.5*iqr_pm25
    std_dev=y1[feature].std()
    return y1.loc[~((y1[feature]<low) | (y1[feature]>upp))]

def outlier_count(df, feature_list):
    """
    function to import data from a json file
    Parameters
    -------------
    filename: 
        name of the file  
    Return
    -------------
    Series or DataFrame:
        The contents of the file are returned in the Series or DataFrame format
    """
    iqr = df[feature_list].quantile(0.75) - df[feature_list].quantile(0.25)
    low = df[feature_list].quantile(0.25) - 1.5*iqr
    up =  df[feature_list].quantile(0.75) + 1.5*iqr
    return len(df.loc[(df[feature_list]<low) | (df[feature_list]>up)])

def get_plot(x, ya, yp, xl):
    """
    This function returns a bokeh plot object
    
    -------------

    Parameters
    x: numpy  aray
        the values on the x-axis
    ya: numpy array
        actual values of dependent variable
    yp: numpy array
        predicted values of dependent variable

    -------------

    Return
    bokeh plot
        Bokeh Plot object where actual and predicted values are plotted against independent variable
    """
    p = figure(width=320, height=320, x_axis_label = xl, y_axis_label = 'AQI', title = f'{xl} and AQI Distribution')
    p.circle(x, ya, size=8, color="navy", alpha=0.9, legend = 'Actual')
    p.square(x, yp, size = 8, color = 'olive', alpha = 0.4, legend = 'Predictions')
    p.legend.location = "top_left"
    return p

def get_grid(plots):
    
    """
    This function returns a bokeh plot object with 2 rows and 5 columns grid
    
    -------------

    Parameters
    plots: 
        list of bokeeh plot objects

    -------------

    Return
    bokeh plot
        bokeh grid object
    """

    r1 = row(plots[0:5])
    r2 = row(plots[5:10])
    
    cols = column([r1, r2])
    return cols

def get_Confusion_matrix(matrix, class_names, model):
    
    """
    This function displays a confusion matrix
    
    -------------

    Parameters
    matrix: python list
        2 dimensional python array
    class_names: list
        List of classes in the dataset
    model: String
        Name of the model

    -------------

    Return
        None
    """

    plt.figure(figsize=(16,7))
    sns.set(font_scale=1.4)
    sns.heatmap(matrix, annot=True, annot_kws={'size':10},
            cmap=plt.cm.Greens, linewidths=0.2)

    # Add labels to the plot
    tick_marks = np.arange(len(class_names))
    tick_marks2 = tick_marks + 0.5
    plt.xticks(tick_marks, class_names, rotation=25)
    plt.yticks(tick_marks2, class_names, rotation=0)
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.title('Confusion Matrix for ' +model)
    plt.show()

def feature_importance(importances, model_name, feature_list):

    """
    This function displays a importances of features in form of a bar graph
    
    -------------

    Parameters

    importances: python list
        List of float or integer values which has importances
    model_name: String
        Name of the model
    feature_list: List of Strings
        List of feature names
    -------------

    Return
        None
    """

    feature_importance = np.array(importances)
    feature_names = np.array(feature_list)

    #Create a DataFrame using a Dictionary
    data={'feature_names':feature_names,'feature_importance':feature_importance}
    fi_df = pd.DataFrame(data)

    #Sort the DataFrame in order decreasing feature importance
    fi_df.sort_values(by=['feature_importance'], ascending=False,inplace=True)

    #Define size of bar plot
    plt.figure(figsize=(10,20))
    #Plot Searborn bar chart
    sns.barplot(x=fi_df['feature_importance'], y=fi_df['feature_names'])
    #Add chart labels
    plt.title(model_name + ' FEATURE IMPORTANCE')
    plt.xlabel('FEATURE IMPORTANCE')
    plt.ylabel('FEATURE NAMES')
