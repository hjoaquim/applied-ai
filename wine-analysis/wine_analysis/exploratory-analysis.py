#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

display = print
# In[2]:


df_red = pd.read_csv(
    "/home/hjoaquim/Documents/applied-ai/wine-analysis/data/red_wine.csv"
)
df_white = pd.read_csv(
    "/home/hjoaquim/Documents/applied-ai/wine-analysis/data/white_wine.csv"
)


# In[3]:


print("Red wine data")
display(df_red.head())
print("Withe wine data")
display(df_white.head())


# # Exploratory Data Analysis

# In[4]:


print(f"Shape: red wine: {df_red.shape}")
print(f"Shape: white wine: {df_white.shape}")


# In[5]:


print("Descriptive Statistics: red wine")
display(df_red.describe())
print("Descriptive Statistics: white wine")
display(df_white.describe())


# In[6]:


print("Types: red wine")
display(df_red.dtypes)
print("Types: white wine")
display(df_white.dtypes)


# In[7]:


df_red.isnull().sum()


# In[8]:


df_white.isnull().sum()


# In[9]:


# Rows with null values
print("Red wine")
display(df_red[df_red["residual sugar"].isnull()])
print("White wine")
display(df_white[df_white["residual sugar"].isnull()])


# In[10]:


# Univariate analysis - understanding the distribution of each variable
print("Red wine")
df_red.hist(figsize=(10, 10))
plt.show()
print("White wine")
df_white.hist(figsize=(10, 10))
plt.show()


# ## Interpreting the Histograms
#
# ### Red wine
#
# 1. **Fixed Acidity**:
#     - The data is somewhat right-skewed, indicating that most wines have a lower fixed acidity, but a few have a high fixed acidity.
# 2. **Volatile Acidit**y:
#     - The distribution appears fairly normal but slightly right-skewed, suggesting that while many wines have a moderate level of volatile acidity, there are fewer wines with high volatile acidity.
# 3. **Citric Acid**:
#     - This histogram shows a right-skewed distribution with a peak close to zero, indicating most wines have low citric acid content. There's a long tail towards higher citric acid contents, showing that fewer wines have higher amounts.
# 4. **Residual Sugar**:
#     - The distribution is heavily right-skewed. Most of the wines have low residual sugar, with very few wines having high residual sugar levels.
# 5. **Chlorides**:
#     - The data is right-skewed, which suggests most wines have low chloride content, with some exceptional wines having higher chloride levels.
# 6. **Free Sulfur Dioxide**:
#     - This variable shows a right-skewed distribution. The majority of wines have a lower amount of free sulfur dioxide, and fewer wines have a higher amount.
# 7. **Total Sulfur Dioxide**:
#     - The distribution is also right-skewed, indicating that most wines have a lower concentration of total sulfur dioxide, but there are some with a significantly higher concentration.
# 8. **Density**:
#     - The density histogram appears to be somewhat normally distributed, but with a peak slightly to the left, suggesting many wines have a density around a common value with fewer wines at the lower and higher ends.
# 9. **pH**:
#     - The distribution looks approximately normal, indicating that the pH levels of most wines are concentrated around a central value, which is typical for wine.
# 10. **Sulphates**:
#     - The histogram for sulphates is right-skewed, indicating most wines have a low to moderate sulphate level, and fewer wines have high levels of sulphates.
# 11. **Alcohol**:
#     - The alcohol content appears to have a bimodal distribution, suggesting two different groups within the dataset, possibly indicating wines with typically lower alcohol content and those with higher alcohol content.
# 12. **Quality**:
#     - The quality variable shows a distribution that is not quite normal and is skewed to the right. It suggests that there are more wines of medium quality and fewer of high quality.
#
# ### White wine
#
# 1. **Fixed Acidity**:
#     - This histogram shows a right-skewed distribution with a peak around 6-8, indicating that most wines have moderate fixed acidity levels, with fewer wines having higher acidity.
# 2. **Volatile Acidity**:
#     - The distribution is slightly right-skewed, with most wines having low to moderate volatile acidity.
# 3. **Citric Acid**:
#     - The data is right-skewed with most wines containing lower levels of citric acid and fewer wines having higher levels.
# 4. **Residual Sugar**:
#     - A heavily right-skewed distribution indicates that most wines have low levels of residual sugar, with a few wines showing much higher levels.
# 5. **Chlorides**:
#     - This histogram shows a right-skewed distribution, with most wines having low chloride content.
# 6. **Free Sulfur Dioxide**:
#     - The distribution is right-skewed, with a concentration of wines having lower levels of free sulfur dioxide.
# 7. **Total Sulfur Dioxide**:
#     - The data shows a right-skewed distribution, indicating that most wines have low to moderate levels of total sulfur dioxide.
# 8. **Density**:
#     - The histogram suggests a normal distribution centered around 0.996 to 0.998, indicating that most wines have a similar density.
# 9. **pH**:
#     - The distribution appears to be normally distributed with a slight left skew, indicating that the pH of most wines is concentrated around a central value.
# 10. **Sulphates**:
#     - The data is right-skewed, with the majority of wines having lower sulphate levels.
# 11. **Alcohol**:
#     - This histogram is quite right-skewed, with a majority of wines having an alcohol content around 9-10%.
# 12. **Quality**:
#     - The quality distribution shows that most wines are of average quality, with few wines at the high and low ends of the quality scale.
#
# ### Summary
#
# **Right Skewness**: Many variables such as fixed acidity, volatile acidity, citric acid, residual sugar, chlorides, free sulfur dioxide, total sulfur dioxide, and sulphates show a right-skewed distribution in both sets of wine data. This indicates a commonality in the distribution where a large number of wines tend to have lower values for these chemical properties, with fewer wines having high values.
#
# **Density**: In both datasets, the density appears to be normally distributed, suggesting a similarity in wine production processes that result in a consistent density.
#
# **pH**: Both histograms for pH are approximately normal, indicating that the acidity level in both sets of wines tends to cluster around a central typical value.
#
# **Alcohol Content**: Both sets have histograms that show skewness, but the second dataset's (**white wine**) histogram is more right-skewed, indicating a tendency for lower alcohol content with fewer wines having higher alcohol content, whereas the first (**red wine**)dataset showed a bimodal distribution.
#
# **Quality**: The quality variable for both datasets is not normally distributed and shows a concentration of data around the median quality levels, suggesting most wines are of average quality in both datasets.
#
# These similarities suggest that despite differences that may exist in the details of each wine's properties, there are common trends in terms of their chemical composition and quality distribution. These trends could reflect industry standards, common winemaking practices, or consumer preferences that influence how wines are produced and marketed.

# In[11]:


# Bar chart of quality counts
red_q = df_red["quality"].value_counts().sort_index()
white_q = df_white["quality"].value_counts().sort_index()
print("Red wine")
red_q.plot(kind="bar", title="Red Wine Quality Counts")
plt.show()
print("White wine")
white_q.plot(kind="bar", title="White Wine Quality Counts")
plt.show()


# In[12]:


# Correlation matrix
print("Red wine")
display(df_red.corr())
plt.figure(figsize=(12, 9))
sns.heatmap(df_red.corr(), annot=True, fmt=".2f")
plt.show()

print("White wine")
display(df_white.corr())
plt.figure(figsize=(12, 9))
sns.heatmap(df_white.corr(), annot=True, fmt=".2f")
plt.show()


# ## Interpret the heatmap
#
# ### Red wine
#
# 1. **Fixed Acidity:**
#     - Has a strong positive correlation with citric acid (0.67), which is expected as both are acids.
#     - Has a strong negative correlation with pH (-0.69), indicating that as fixed acidity increases, pH tends to decrease (become more acidic).
#
# 2. **Volatile Acidity:**
#     - Shows a strong negative correlation with citric acid (-0.56), which suggests that as volatile acidity increases, citric acid tends to decrease.
#     - Has a moderately negative correlation with quality (-0.38), implying that higher volatile acidity may be associated with lower wine quality.
#
# 3. **Citric Acid:**
#     - Has a moderate positive correlation with sulphates (0.31), indicating a relationship where wines with more citric acid also tend to have more sulphates.
#
# 4. **Residual Sugar, Chlorides, Free Sulfur Dioxide, Total Sulfur Dioxide:**
#     - These variables do not show strong correlations with other variables in the heatmap (most coefficients are close to 0), suggesting that they do not have strong linear relationships with other measured components in red wine.
#
# 5. **Density:**
#     - Has a moderate positive correlation with fixed acidity (0.67) and a moderate negative correlation with alcohol (-0.50), which makes sense as alcohol tends to be less dense than water.
#
# 6. **pH:**
#     - Shows a strong negative correlation with fixed acidity (-0.69) and a moderate negative correlation with citric acid (-0.54), which aligns with the chemical understanding that more acid lowers the pH value.
#
# 7. **Sulphates:**
#     - Has a moderate positive correlation with citric acid (0.31) and a mild positive correlation with quality (0.25), suggesting that sulphates might contribute positively to the perceived quality of red wine.
#
# 8. **Alcohol:**
#     - Exhibits the strongest positive correlation with quality (0.47) within this heatmap, suggesting that higher alcohol content may be associated with higher quality ratings in red wine.
#
# 9. **Quality:**
#     - Aside from alcohol, it has a moderate negative correlation with volatile acidity (-0.38) and a mild positive correlation with sulphates (0.25). This suggests that wines with less volatile acidity and more sulphates tend to be rated as higher quality.
#
# ### White wine
#
# 1. **Fixed Acidity:**
#     - Has a moderate negative correlation with pH (-0.43), suggesting that as fixed acidity increases, pH tends to decrease.
#     - There are no strong positive correlations with any other variables.
#
# 2. **Volatile Acidity:**
#     - Does not show strong correlations with any of the variables. The highest correlation is a very weak negative correlation with quality (-0.19), indicating a slight tendency for lower quality with higher volatile acidity.
#
# 3. **Citric Acid:**
#     - Also shows no strong correlations with other variables in the dataset.
#
# 4. **Residual Sugar:**
#     - Has a strong positive correlation with density (0.84), which is logical since sugar increases the density of the liquid.
#     - There's a moderate positive correlation with total sulfur dioxide (0.39). This might be due to the fact that wines with higher sugar content could need more sulfur dioxide as a preservative.
#
# 5. **Chlorides:**
#     - No strong correlations are observed with chlorides and other variables.
#
# 6. **Free Sulfur Dioxide and Total Sulfur Dioxide:**
#     - Show a strong positive correlation with each other (0.62). This is expected as free sulfur dioxide is a part of the total sulfur dioxide measurement.
#     - Both have a moderate positive correlation with residual sugar (0.29 for free and 0.39 for total), possibly indicating that sweeter wines might contain higher levels of sulfur dioxide, possibly due to the preservative qualities needed for sweeter wines.
#
# 7. **Density:**
#     - Shows a strong positive correlation with residual sugar (0.84) as mentioned, and a moderate positive correlation with alcohol (0.52). This latter correlation could be due to higher sugar content wines undergoing more fermentation, leading to higher alcohol content.
#
# 8. **pH:**
#     - Shows a moderate negative correlation with fixed acidity (-0.43), which is a typical chemical relationship (acidic substances lower the pH).
#
# 9. **Sulphates:**
#     - There is a weak correlation between sulphates and other variables in the dataset.
#
# 10. **Alcohol:**
#     - Has a moderate negative correlation with density (-0.60), which makes sense because alcohol is less dense than water.
#     - There is a moderate positive correlation with quality (0.35), suggesting that wines with higher alcohol content might be associated with higher quality.
#
# 11. **Quality:**
#     - The strongest correlation with quality is with alcohol (0.35), indicating that as alcohol content increases, quality rating might also increase.
#     - Quality shows weak negative correlations with volatile acidity (-0.19) and density (-0.30), suggesting that higher quality wines might have lower volatile acidity and density.

# ## Outliers
#
# **IQR (Interquartile Range)**: The IQR is the difference between the 75th percentile (Q3) and the 25th percentile (Q1) in a dataset. An outlier is often considered to be any data point that lies more than 1.5 times the IQR above the third quartile or below the first

# In[13]:


Q1 = df_red.quantile(0.25)
Q3 = df_red.quantile(0.75)
IQR = Q3 - Q1
outliers = df_red[(df_red < (Q1 - 1.5 * IQR)) | (df_red > (Q3 + 1.5 * IQR))]
display(outliers)
display(outliers.notna().sum())


# In[14]:


Q1 = df_white.quantile(0.25)
Q3 = df_white.quantile(0.75)
IQR = Q3 - Q1
outliers = df_white[(df_white < (Q1 - 1.5 * IQR)) | (df_white > (Q3 + 1.5 * IQR))]
display(outliers)
display(outliers.notna().sum())
