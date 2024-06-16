# Facebook Ad Campaigns Analysis and Visualization

## Overview
This project demonstrates a comprehensive process of analyzing and visualizing Facebook Ad Campaign data using Python and popular data science libraries such as **Pandas**, **Seaborn**, and **Matplotlib**. The analysis includes calculating moving averages, aggregating data, and identifying correlations to provide insights into advertising spend and Return on Marketing Investment (ROMI) for better decision-making and analysis.

## Advanced Techniques and Approaches Used

- **Data Cleaning**: Handling NaN values to ensure data integrity and accuracy in analysis.
- **Rolling Averages**: Utilizing the `rolling()` method to calculate moving averages for smoothing time series data, providing clearer trends.
- **Data Aggregation**: Grouping data by different categories to compute aggregate statistics, which simplifies complex data and highlights key trends.
- **Visualization**: Creating various plots (line plots, bar plots, box plots, histograms, scatter plots) to visually represent data trends, distributions, and relationships.
- **Heatmap Analysis**: Building a heatmap to visualize the correlation between different numerical indicators, aiding in the identification of key relationships.

## Conclusion

This analysis of Facebook Ad Campaigns provided valuable insights into advertising spend and ROMI across different campaigns. Key findings include:

- The rolling average analysis highlighted fluctuations in daily advertising spend and ROMI throughout 2021.
- The total spend and ROMI analysis per campaign revealed which campaigns were the most cost-effective and yielded the highest returns.
- The distribution of daily ROMI and the overall ROMI distribution provided a comprehensive view of performance variability across campaigns.
- Correlation analysis identified strong relationships between various metrics, particularly between total spend and total value, aiding in a better understanding of what drives value.

These insights can guide future ad spending decisions to optimize return on investment.

## Python Code

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

plt.rcParams['figure.figsize'] = [30, 10]  # setting standard plot size for all notebook
sns.set(style="whitegrid")  # general style of the plots

# Load data
campaigns = pd.read_csv('C:/Users/plish/Desktop/Python_HW/Facebook Ad campaigns Analysis and Viz/facebook_ads_data.csv')

# Cleaning data from NaN values (strings)
campaigns.dropna(subset=['cpc', 'ctr', 'romi'], inplace=True)
campaigns.info()  # data structure check

# Create grouped data set for the year 2021
data_2021 = campaigns.loc[campaigns['ad_date'].between('2021-01-01', '2021-12-31')]
data_set = data_2021.groupby('ad_date').agg({'total_spend': 'sum', 'romi': 'mean'})

# Daily rolling average Ad Spend in 2021 Chart
sns.lineplot(x='ad_date', y=data_2021['total_spend'].rolling(10).mean(), data=data_2021, color='orange')
plt.title('Daily Rolling Average Ad Spend in 2021', fontdict={'fontsize': 30, 'fontweight': 'bold'}, pad=20)
plt.xlabel('Date', fontsize='xx-large')
plt.ylabel('Ad Spend, $', fontsize='xx-large')
plt.xticks(rotation=90)
plt.xticks(list(campaigns['ad_date'])[1::30])
plt.show()

# Daily ROMI in 2021 Chart
sns.lineplot(x='ad_date', y=data_2021['romi'].rolling(10).mean(), data=data_2021, color='green')
plt.title('Daily Rolling Average ROMI in 2021', fontdict={'fontsize': 30, 'fontweight': 'bold'}, pad=20)
plt.xlabel('Date', fontsize='xx-large')
plt.ylabel('ROMI', fontsize='xx-large')
plt.xticks(rotation=90)
plt.xticks(list(campaigns['ad_date'])[1::30])
plt.show()

# Creating data frame aggregated by ad campaign
campaign_data = campaigns.groupby('campaign_name').agg({'total_spend': 'sum', 'romi': 'mean'}).reset_index()

# Total Spend on Ad Campaign
ax = sns.barplot(x='campaign_name', y='total_spend', data=campaign_data)
plt.title('Total Ad Spend by Campaigns', fontdict={'fontsize': 30, 'fontweight': 'bold'}, pad=20)
plt.xlabel(None, fontsize='xx-large')
plt.ylabel('Total Spend', fontsize='xx-large')

# Display values on each bar
for p in ax.patches:
    ax.annotate(f'{p.get_height():.0f}', (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center', fontsize=16, color='black', xytext=(0, 10), textcoords='offset points')
plt.show()

# Total ROMI on each Campaign
ax = sns.barplot(x='campaign_name', y='romi', data=campaign_data)
plt.title('ROMI by Campaigns', fontdict={'fontsize': 30, 'fontweight': 'bold'}, pad=20)
plt.xlabel(None, fontsize='xx-large')
plt.ylabel('ROMI', fontsize='xx-large')

# Display values on each bar
for p in ax.patches:
    ax.annotate(f'{(p.get_height())*100:.2f}%', (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center', fontsize=16, color='black', xytext=(0, 10), textcoords='offset points')
ax.tick_params(axis='x', which='major', labelsize=16)
plt.show()

# Determine the daily ROMI spread in each campaign (by campaign name) using a box plot
from matplotlib.ticker import PercentFormatter  # import to be able to format Y-axis to %
campaigns['romi_perc'] = campaigns['romi'] * 100  # create new column to convert ROMI to %
daily_romi = sns.boxplot(x='campaign_name', y='romi_perc', data=campaigns)
plt.title('Distribution of Daily ROMI by Campaign', fontdict={'fontsize': 30, 'fontweight': 'bold'}, pad=20)
plt.xlabel(None, fontsize='xx-large')
plt.ylabel('ROMI (%)', fontsize='xx-large')
daily_romi.set_ylim(0, 275)  # set y-axis limits
daily_romi.yaxis.set_major_formatter(PercentFormatter(decimals=0))
daily_romi.tick_params(axis='x', which='major', labelsize=16)
plt.show()

# Create histogram to show the distribution of ROMI values in the data set table.
hist_plot = sns.histplot(campaigns['romi_perc'], bins=30, color='skyblue', edgecolor='black', kde=True)
plt.title('Histogram of ROMI Values Distribution', fontdict={'fontsize': 30, 'fontweight': 'bold'}, pad=20)
plt.xlabel('ROMI (%)', fontsize='xx-large')
plt.ylabel('Frequency', fontsize='xx-large')
hist_plot.tick_params(axis='both', which='major', labelsize=16)
plt.xticks(range(0, int(campaigns['romi_perc'].max()) + 50, 10))  # set x-axis tick step
hist_plot.set_xlim(0, 270)  # set x-axis limits
hist_plot.set_ylim(0, 130)  # set y-axis limits
plt.show()

# Build a heatmap of the correlation
df = campaigns.iloc[:, 2:9]  # get data frame for the chart
plt.figure(figsize=(30, 12))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', xticklabels=['Total Spend', 'Total Impressions', 'Total Clicks', 'Total Value', 'CPC', 'CPM', 'CTR'], yticklabels=['Total Spend', 'Total Impressions', 'Total Clicks', 'Total Value', 'CPC', 'CPM', 'CTR'], linewidth=1)
plt.title('Сorrelation Heatmap', fontdict={'fontsize': 20, 'fontweight': 'bold'}, pad=10)
plt.show()

# What is "total_value" correlated with?
correlation_matrix = df.corr()
# Drop self-correlation
total_value_corr = correlation_matrix['total_value'].drop('total_value')
print(f'Correlation of "Total value" with other indicators:\n{round(total_value_corr,2)}')

# Which indicators have the highest and lowest correlation?
# Get the highest correlation pairs
highest_corr = correlation_matrix.unstack().sort_values().drop_duplicates().tail(2)
# Get the lowest correlation pairs
lowest_corr = correlation_matrix.unstack().sort_values().drop_duplicates().head(2)
print(f'Highest correlation:\n{highest_corr}\n')
print(f'Lowest correlation:\n{lowest_corr}')

# Create a scatter plot with linear regression
sns.lmplot(x='total_spend', y='total_value', data=campaigns)
plt.title('Scatter plot with a Linear Regression Line', fontdict={'fontsize': 15, 'fontweight': 'bold'}, pad=20)
plt.xlabel('Total Spend', fontsize='medium')
plt.ylabel('Total Value', fontsize='medium')
hist_plot.tick_params(axis='both', which='major', labelsize=10)
hist_plot.set_xlim(0, 2500)  # set x-axis limits
hist_plot.set_ylim(0, 3000)  # set y-axis limits
plt.show()
