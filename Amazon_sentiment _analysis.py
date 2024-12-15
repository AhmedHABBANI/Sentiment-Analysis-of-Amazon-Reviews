import numpy as np
import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import re
from textblob import TextBlob
from wordcloud import WordCloud
import seaborn as sns  # Corrected here, 'seaborn' instead of 'Seaborn'
import matplotlib.pyplot as plt
import cufflinks as cf
%matplotlib inline
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
cf.go_offline()
import plotly.graph_objs as go
from plotly.subplots import make_subplots

import warnings
warnings.filterwarnings("ignore")
warnings.warn("this will not show")

# Set Pandas options to display all columns
pd.set_option('display.max_columns', None)

# Load dataset from a CSV file
df = pd.read_csv("/content/sample_data/amazon.csv")

# Sort the DataFrame by the 'wilson_lower_bound' column in descending order
df = df.sort_values("wilson_lower_bound", ascending=False)

# Check if the 'Unnamed : 0' column exists, then drop it
if 'Unnamed : 0' in df.columns:
    df.drop('Unnamed : 0', inplace=True, axis=1)

# Display the first few rows of the DataFrame
df.head()

# Define a function to analyze missing values in the DataFrame
def missing_values_analysis(df):
    # Find columns with missing values
    na_columns_ = [col for col in df.columns if df[col].isnull().sum() > 0]

    # Number of missing values per column
    n_miss = df[na_columns_].isnull().sum().sort_values(ascending=True)

    # Percentage of missing values per column
    ratio = (df[na_columns_].isnull().sum() / df.shape[0] * 100).sort_values(ascending=True)

    # Create a DataFrame to display the results
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['Missing Values', 'Ratio (%)'])
    return missing_df

# Define a function to check the general properties of the DataFrame
def check_dataframe(df, head=5, tail=5):
    print("SHAPE".center(82, '~'))
    print('Rows: {}'.format(df.shape[0]))
    print('Columns: {}'.format(df.shape[1]))

    print("TYPES".center(82, '~'))
    print(df.dtypes)  # Corrected here: df.dtypes instead of df.types

    print("MISSING VALUES".center(82, '~'))
    print(missing_values_analysis(df))

    print("DUPLICATED VALUES".center(83, '~'))
    print('Duplicated rows: {}'.format(df.duplicated().sum()))

    print("QUANTILES".center(82, '~'))
    # Filter only numeric columns
    numeric_df = df.select_dtypes(include=[np.number])
    print(numeric_df.quantile([0, 0.05, 0.5, 0.95, 0.99, 1]).T)

    print("HEAD".center(82, '~'))
    print(df.head(head))

    print("TAIL".center(82, '~'))
    print(df.tail(tail))

# Call the function to check the DataFrame
check_dataframe(df)

# Define a function to check unique values and their counts in each column
def check_class(dataframe):
    nunique_df = pd.DataFrame({
        'Variable': dataframe.columns,
        'Classes': [dataframe[col].unique() for col in dataframe.columns]
    })
    nunique_df = nunique_df.sort_values(by='Classes', key=lambda x: x.map(len), ascending=False)
    nunique_df = nunique_df.reset_index(drop=True)
    return nunique_df

# Call the function to check the unique values in each column
check_class(df)

# Define a function to create categorical variable summary plots
constraints = ['#FF5733', '#33FF57', '#3357FF', '#F1C40F', '#8E44AD']
def categorical_variable_summary(df, column_name):
    # Create a figure with subplots
    fig = make_subplots(rows=1, cols=2, 
                        subplot_titles=("Countplot", "Percentages"),
                        specs=[[{"type": "xy"}, {"type": "domain"}]])

    # Plot the countplot (bar chart)
    fig.add_trace(go.Bar(
        y=df[column_name].value_counts().values.tolist(),
        x=[str(i) for i in df[column_name].value_counts().index],
        text=df[column_name].value_counts().values.tolist(),
        textfont=dict(size=15),
        name=column_name,
        textposition='auto',
        showlegend=False,
        marker=dict(color=constraints, 
                    line=dict(color='#DBE6EC', width=1))),
        row=1, col=1
    )

    # Plot the percentages (pie chart)
    fig.add_trace(go.Pie(
        labels=df[column_name].value_counts().keys(),
        values=df[column_name].value_counts().values,
        textfont=dict(size=20),
        textposition='auto',
        showlegend=False,
        name=column_name,
        marker=dict(colors=constraints)),
        row=1, col=2
    )

    # Update the layout of the figure
    fig.update_layout(
        title=dict(
            text=column_name,
            y=0.9,
            x=0.5,
            xanchor='center',
            yanchor='top'
        ),
        template='plotly_white'
    )

    # Display the figure
    iplot(fig)

# Call the function to plot categorical variable summary for the 'overall' column
categorical_variable_summary(df, 'overall')

# Clean the review text by removing non-alphabetic characters
df.reviewText.head()
review_example = re.sub("[^a-zA-Z]",'',review_example)
review_example = review_example.lower().split()
rt = lambda x : re.sub("[^a-zA-Z]",' ',str(x))
df["reviewText"]= df["reviewText"].map(rt)
df["reviewText"]= df["reviewText"].str.lower()

# Apply TextBlob to get polarity and subjectivity
df[['polarity', 'subjectivity']] = df['reviewText'].apply(lambda text: pd.Series(TextBlob(text).sentiment))

# Initialize the sentiment analyzer from VADER
analyzer = SentimentIntensityAnalyzer()

# Analyze the sentiment of each review
for index, row in df.iterrows():
    review_text = row['reviewText']
    score = analyzer.polarity_scores(review_text)
    
    # Extract sentiment scores
    neg = score['neg']
    neu = score['neu']
    pos = score['pos']
    
    # Determine the sentiment
    if neg > pos:
        df.loc[index, 'sentiment'] = "Negative"
    elif pos > neg:
        df.loc[index, 'sentiment'] = "Positive"
    else:
        df.loc[index, 'sentiment'] = "Neutral"

# Display the results (review text with sentiment scores)
print(df[['reviewText', 'polarity', 'subjectivity', 'sentiment']].head())

# Display the top 10 positive reviews sorted by 'wilson_lower_bound'
df[df['sentiment']=='Positive'].sort_values("wilson_lower_bound", ascending=False).head(10)
