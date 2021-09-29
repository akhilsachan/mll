from flask import Flask,render_template,request,redirect, url_for

import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# HTML ----------------> PYTHON
#request: Allows you to request data

app = Flask(__name__)

def predict_attr(data):
    
    attrition = pd.read_csv('train.csv')
    attrition['Attrition'] = np.where(attrition.Attrition == "Yes", 1,0)
 	
    attrition = attrition.drop(['DailyRate', 'HourlyRate','PerformanceRating', 
                                'MonthlyRate', 'StandardHours','Over18','EmployeeCount',
                                'EmployeeNumber'], axis=1)
         
    with open('BEST_MODEL.pkl', 'rb') as handle:
        model = pickle.load(handle)
    
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

# first take default page
@app.route('/')

def index():
    return render_template('index.html')

# Which page to open after index.html?
# if '/about' in url, then go to about else open this page

@app.route('/predict',methods=['POST'])
def predict():
    if request.method == 'POST':
        
        dict_info = request.form
        
        pred = predict_attr(dict_info)
        
        print('PRED:',pred)
		
        lst = []
        
        if pred[0] == 1:
            lst.append('will')
            lst.append('1F61F')
        else:
            lst.append('will not')
            lst.append('1F642')
		
        return render_template('result.html',prediction = lst)

if __name__ == '__main__':
    app.run(debug=True)