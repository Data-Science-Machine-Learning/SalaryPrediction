# DESCRIPTION
# This is example of Linear Regression model for Salary prediction,
# Inputs: Location, Age, Experience, Qualification, Job title
# Output: Salary (prediction based on Inputs)

# INPUT CODES FOR CATEGORICAL INPUTS (THIS COMES FROM ONE HOT ENCODING)
# city:
# 1,0,0 - Ahmedabad
# 0,1,0 - Baroda
# 0,0,1 - Mumbai
# 0,0,0 - Pune
#
# qualification:
# 1 = bachelors
# 0 = masters
#
# job title:
# 1,0,0 - ai developer
# 0,1,0 - mobile developer
# 0,0,1 - voip developer
# 0,0,0 - web developer

# PREDICT IN THIS ORDER OF INPUT
# sample - age, experience, city, qualification, job title

import pandas as pd
from sklearn import linear_model

# defining linear regression model
reg = linear_model.LinearRegression()

# one hot encoding for Categorical variables
df = pd.read_csv('salary.csv')  # loading data from csv

# getting dummy variables for categorical input variables
dummies_city = pd.get_dummies(df.city)
dummies_qualification = pd.get_dummies(df.qualification)
dummies_job_title = pd.get_dummies(df.job_title)

# concat original data frame with dummies
merged = pd.concat([df, dummies_city, dummies_qualification, dummies_job_title], axis="columns")

# dropping categorical variables and last column of each categorical variable
final = merged.drop(["city", "qualification", "job_title", "Pune", "masters", "webdeveloper"], axis="columns")

# deriving input-x and output-y variables
x = final.drop(["salary"], axis="columns")
y = final.salary

# training model with all data
reg.fit(x, y)

score = reg.score(x, y)  # checking accuracy score of this model
print('Model Accuracy Score: ' + str(score))

# predicting salary for various input samples
# example - age, experience, city, qualification, job title. The city, qualification and job title codes are shown above
predict = reg.predict([[27, 4, 1, 0, 0, 0, 0, 0, 1]])
print('Predicted salary for Abad loc. for Ag-27, Exp.-4, Qua.-masters, Job title-VoIPDeveloper: ' + str(predict))
predict = reg.predict([[33, 9, 1, 0, 0, 1, 0, 0, 1]])
print('Predicted salary for Abad loc. for Ag-33, Exp.-9, Qua.-bachelors, Job title-VoIPDeveloper: ' + str(predict))

predict = reg.predict([[29, 6, 0, 1, 0, 0, 0, 0, 1]])
print('Predicted salary for Baroda loc. for Ag-29, Exp.-6, Qua.-masters, Job title-VoIPDeveloper: ' + str(predict))

predict = reg.predict([[29, 6, 0, 0, 1, 0, 0, 0, 1]])
print('Predicted salary for Mumbai loc. for Ag-29, Exp.-6, Qua.-masters, Job title-VoIPDeveloper: ' + str(predict))

predict = reg.predict([[27, 4, 0, 0, 0, 0, 0, 0, 1]])
print('Predicted salary for Pune loc. for Ag-27, Exp.-4, Qua.-masters, Job title-VoIPDeveloper: ' + str(predict))
predict = reg.predict([[33, 9, 0, 0, 0, 1, 0, 0, 1]])
print('Predicted salary for Pune loc. for Ag-33, Exp.-9, Qua.-bachelors, Job title-VoIPDeveloper: ' + str(predict))
