# MachineLearning
This is Linear Regression model to predict Salary for different developers roles working at different Geo location based on age and experience and qualification.

Here, the salary.csv file is the original data we have. It contains city, age, experience, qualification, job title, salary fields. In that city, qualification and job title are categorical variables, that we may need to convert in numeric values as our Linear Regression model which we are going to use works with numeric values only and takes numeric values. For that we are using one hot encoding method. So, for that we used get_dummies method to get numeric values of these categorical variables and concated to original data frame, then we dropping those original categorical variables and last column of each categorical variable as a thumb rule. So, now we have final.csv is the actual processed data on which we will apply predication model. I have given merged.csv and final.csv files just to see and understand data frame at various level of data processing. Inside code 'final' variable holding processed data which we will feed in our model for prediction. x is input data and y is expected output (y = salary prediction). Inside code I have printed accuracy score of our prediction model and some predicted values of exprected salary.

Below are some outputs for predictions:

Model Accuracy Score: 0.9482808377308707
Predicted salary for Abad loc. for Ag-27, Exp.-4, Qua.-masters, Job title-VoIPDeveloper: [39552.08333333]
Predicted salary for Abad loc. for Ag-33, Exp.-9, Qua.-bachelors, Job title-VoIPDeveloper: [49046.875]
Predicted salary for Baroda loc. for Ag-29, Exp.-6, Qua.-masters, Job title-VoIPDeveloper: [41614.58333333]
Predicted salary for Mumbai loc. for Ag-29, Exp.-6, Qua.-masters, Job title-VoIPDeveloper: [47656.25]
Predicted salary for Pune loc. for Ag-27, Exp.-4, Qua.-masters, Job title-VoIPDeveloper: [47260.41666667]
Predicted salary for Pune loc. for Ag-33, Exp.-9, Qua.-bachelors, Job title-VoIPDeveloper: [56755.20833333]

- salary_prediction.py is the main code file which predicts salary for the given criteria.
- salary.csv (Actual Data)
- merged.csv (concated dummy variables with original data frame)
- final.csv (Processed data, ready to apply in prediction model)
