# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def predict_attr(data, model):
    
    attrition = pd.read_csv('train.csv')
    attrition['Attrition'] = np.where(attrition.Attrition == "Yes", 1,0)
 	
    attrition = attrition.drop(['DailyRate', 'HourlyRate','PerformanceRating', 
                                'MonthlyRate', 'StandardHours','Over18','EmployeeCount',
                                'EmployeeNumber'], axis=1)
    
    test = pd.DataFrame.from_dict(data.items()).set_index(0).T
    test = test.drop(['Submit','Emp_no'],axis=1)
    test['Attrition'] = 1
    
    attrition['train'] = 1
    test['train'] = 0
    
    for each in attrition.columns:
        test[each] = test[each].astype(attrition[each].dtype)
    
    continuous = ['Age', 'YearsAtCompany', 'MonthlyIncome','NumCompaniesWorked',
                  'PercentSalaryHike','DistanceFromHome','TotalWorkingYears',
                  'TrainingTimesLastYear','YearsInCurrentRole','YearsSinceLastPromotion',
                  'YearsWithCurrManager']
    
    test['Age'] = int(np.interp(test['Age'],[1,60], [attrition['Age'].min(),attrition['Age'].max()]))
    
    test['DistanceFromHome'] = int(np.interp(test['DistanceFromHome'],[1,40], 
                                    [attrition['DistanceFromHome'].min(),attrition['DistanceFromHome'].max()]))
    
    test['YearsAtCompany'] = int(np.interp(test['YearsAtCompany'],[1,40], 
                                    [attrition['YearsAtCompany'].min(),attrition['YearsAtCompany'].max()]))
    
    test['MonthlyIncome'] = int(np.interp(test['MonthlyIncome'],[10000,200000], 
                                    [attrition['MonthlyIncome'].min(),attrition['MonthlyIncome'].max()]))
    
    test['NumCompaniesWorked'] = int(np.interp(test['NumCompaniesWorked'],[1,9], 
                                    [attrition['NumCompaniesWorked'].min(),attrition['NumCompaniesWorked'].max()]))
    
    test['PercentSalaryHike'] = int(np.interp(test['PercentSalaryHike'],[1,25], 
                                    [attrition['PercentSalaryHike'].min(),attrition['PercentSalaryHike'].max()]))
    
    test['TotalWorkingYears'] = int(np.interp(test['TotalWorkingYears'],[1,40], 
                                    [attrition['TotalWorkingYears'].min(),attrition['TotalWorkingYears'].max()]))
    
    test['TrainingTimesLastYear'] = int(np.interp(test['TrainingTimesLastYear'],[1,9], 
                                    [attrition['TrainingTimesLastYear'].min(),attrition['TrainingTimesLastYear'].max()]))
    
    test['YearsInCurrentRole'] = int(np.interp(test['YearsInCurrentRole'],[1,20], 
                                    [attrition['YearsInCurrentRole'].min(),attrition['YearsInCurrentRole'].max()]))
    
    test['YearsSinceLastPromotion'] = int(np.interp(test['YearsSinceLastPromotion'],[1,15], 
                                    [attrition['YearsSinceLastPromotion'].min(),attrition['YearsSinceLastPromotion'].max()]))
    
    test['YearsWithCurrManager'] = int(np.interp(test['YearsWithCurrManager'],[1,20], 
                                    [attrition['YearsWithCurrManager'].min(),attrition['YearsWithCurrManager'].max()]))
    
    combined = pd.concat([attrition,test])
        
    com_num = pd.concat([combined[continuous],combined[['train','Attrition']]],axis=1)
    
    cat = ['Education','EnvironmentSatisfaction',
        'JobInvolvement','JobLevel','JobSatisfaction',
        'RelationshipSatisfaction','BusinessTravel',
        'Department','EducationField','Gender','JobRole','MaritalStatus',
        'OverTime','WorkLifeBalance','StockOptionLevel']
    
    for var in cat:
        combined[var] = combined[var].astype('object')
    
    com_cat = pd.get_dummies(combined[cat])
    
    combined = pd.concat([com_cat,com_num],axis=1)
    
    sc = StandardScaler()
    combined = sc.fit_transform(combined.drop(['Attrition','train'],axis=1))
    
    new_arr = combined[-1]
    new_arr = np.array(new_arr,ndmin=2)
    
    prediction = (model.predict(new_arr))
    # print(model.transform(new_arr))
    
    return prediction

