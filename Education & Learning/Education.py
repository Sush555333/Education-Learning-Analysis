import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

#Load the dataset
df = pd.read_csv('student_engagement_dataset.csv')
'''print(df.head())
print(df.info())
print(df.describe())

 #Check for missing values
print(df.isnull().sum())

# Check for duplicates
print(df.duplicated().sum())

# Remove duplicates
df = df.drop_duplicates()
print(df.duplicated().sum())

df.to_csv ("cleaned_student_engagement.csv", index=False)
print(df)

#Verify if missing values are filled
missing_values_after = df.isnull().sum()
print("Missing values after filling:")
print(missing_values_after)'''


#Dropout Prediction in Online Courses 
#Use engagement metrics to predict and prevent student dropouts from online courses.
#EDA analysis
#Univariate Analysis
plt.figure(figsize=(7, 6))
sns.countplot(x='dropped_out', data=df, palette='viridis')
plt.title('Distribution of Dropouts')
plt.xlabel('dropped_out')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.tight_layout()
plt.legend()
plt.show()

#Bivariate Analysis
#Visualize the relationship between avg_session_duration and dropout rates
plt.figure(figsize=(10, 6))
sns.boxplot(x='avg_session_duration', y='dropped_out', data=df, palette='viridis')
plt.title('Dropout vs avg_session_duration')
plt.xlabel('avg_session_duration')
plt.ylabel('dropped_out')
plt.xticks(rotation=0)
plt.tight_layout()
plt.legend()
plt.show()

#Multivariate Analysis
#Visualize the relationship between multiple avg_session_duration, videos_watched, and dropout rates
plt.figure(figsize=(10, 6))
sns.scatterplot(x='avg_session_duration', y='videos_watched', hue='dropped_out', data=df, palette='viridis')
plt.title('Dropout vs avg_session_duration and videos_watched')
plt.xlabel('avg_session_duration')
plt.ylabel('videos_watched')
plt.xticks(rotation=0)
plt.tight_layout()
plt.legend()
plt.show()


#Correlation matrix heatmap (only for numeric columns)
plt.figure(figsize=(12, 8))
correlation_matrix = df.select_dtypes(include=[np.number]).corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
plt.tight_layout()
plt.show()


#Linear Reggression
#Split the data into features and target variable
X= df[['videos_watched']]
y=df['final_grade']

#Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Create a linear regression model
model = LinearRegression()

#Train the model
model.fit(X_train, y_train)

#Make predictions
y_pred = model.predict(X_test)

print(f"Predicted Rating for test data: {y_pred}")

#Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")

#Visualize the distribution of final grades
plt.figure(figsize=(10, 6))
sns.histplot(df['final_grade'], bins=20, kde=True, color='blue')
plt.title('Distribution of Final Grades')
plt.xlabel('final_grade')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.tight_layout()
plt.legend()
plt.show()


#plot actual vs predicted
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='blue')
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.title('True vs Predicted Performance')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
plt.xlim(y.min(), y.max())
plt.legend()
plt.show()

#Predicting performance for a new student
# Assuming the new student has final_grade of 0.8
new_student = np.array([[0.8]])
predicted_performance = model.predict(new_student)
print(f"Predicted performance for the new student: {predicted_performance[0]}")

#Visualize the regression line
plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='blue', label='Actual')
plt.plot(X, model.predict(X), color='red', label='Regression Line')
plt.xlabel("Videos Watched")
plt.ylabel("Final Grade")
plt.title("Simple Linear Regression")
plt.legend()
plt.show()







