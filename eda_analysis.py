import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set aesthetic style for charts
sns.set_theme(style="whitegrid")

# Create a directory for visualizations if it doesn't exist
output_dir = "visualizations"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Load the cleaned dataset
df = pd.read_csv('train_cleaned.csv')

# Ensure date columns are in datetime format
df['Order Date'] = pd.to_datetime(df['Order Date'])

# 1. Basic Statistics for Sales
print("--- Basic Statistics for Sales ---")
stats = df['Sales'].describe()
print(stats)
print(f"Median: {df['Sales'].median()}")
print("-" * 35)

# 2. Sales Distribution (Histogram)
plt.figure(figsize=(10, 6))
sns.histplot(df['Sales'], bins=50, kde=True, color='royalblue')
plt.title('Distribution of Sales')
plt.xlabel('Sales Amount')
plt.ylabel('Frequency')
plt.xlim(0, 5000) # Limiting x-axis to see the main distribution better
plt.savefig(os.path.join(output_dir, 'sales_distribution.png'))
plt.close()

# 3. Outlier Detection (Boxplot)
plt.figure(figsize=(10, 6))
sns.boxplot(x=df['Sales'], color='salmon')
plt.title('Boxplot for Sales (Outlier Detection)')
plt.xlabel('Sales Amount')
plt.savefig(os.path.join(output_dir, 'sales_boxplot.png'))
plt.close()

# 4. Sales by Category (Bar Chart)
plt.figure(figsize=(10, 6))
category_sales = df.groupby('Category')['Sales'].sum().reset_index().sort_values(by='Sales', ascending=False)
sns.barplot(x='Category', y='Sales', data=category_sales, palette='viridis')
plt.title('Total Sales by Category')
plt.xlabel('Category')
plt.ylabel('Total Sales')
plt.savefig(os.path.join(output_dir, 'sales_by_category.png'))
plt.close()

# 5. Sales Trend Over Time (Line Graph)
# Aggregate sales by month
df['Month_Year'] = df['Order Date'].dt.to_period('M')
monthly_sales = df.groupby('Month_Year')['Sales'].sum().reset_index()
monthly_sales['Month_Year'] = monthly_sales['Month_Year'].dt.to_timestamp()

plt.figure(figsize=(14, 7))
sns.lineplot(x='Month_Year', y='Sales', data=monthly_sales, marker='o', color='darkorange')
plt.title('Monthly Sales Trend (2015-2018)')
plt.xlabel('Order Date')
plt.ylabel('Total Sales')
plt.xticks(rotation=45)
plt.savefig(os.path.join(output_dir, 'sales_trend.png'))
plt.close()

# 6. Correlation (Numerical analysis)
# Selecting numerical columns (Row ID and Postal Code are excluded as they are indices/categorical)
print("\n--- Correlation Analysis ---")
numerical_df = df[['Sales']].copy()
# Let's add 'Order Year' to see if there's a correlation with time
numerical_df['Order_Year'] = df['Order Date'].dt.year
correlation = numerical_df.corr()
print(correlation)
print("-" * 35)

# 7. Pattern Discovery: Sales by Region
plt.figure(figsize=(10, 6))
region_sales = df.groupby('Region')['Sales'].sum().reset_index().sort_values(by='Sales', ascending=False)
sns.barplot(x='Region', y='Sales', data=region_sales, palette='magma')
plt.title('Total Sales by Region')
plt.xlabel('Region')
plt.ylabel('Total Sales')
plt.savefig(os.path.join(output_dir, 'sales_by_region.png'))
plt.close()

print(f"\nEDA Complete. Visualizations saved to the '{output_dir}' directory.")
