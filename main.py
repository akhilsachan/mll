# -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 00:11:26 2020

@author: Aditya
"""
import pickle # to load model
from flask import Flask, request, render_template
from ml_model import predict_attr

app = Flask("Employee Attrition")

# first take default page
@app.route('/')
def index():
    return render_template('index.html')

# Random route to base URL.. GET request(Cannot send data!)
@app.route('/predict',methods=['POST'])
def predict():
    if request.method == 'POST':
        
        data = request.form
        
        with open('model.pkl', 'rb') as handle:
            model = pickle.load(handle)
            handle.close()
        
        pred = predict_attr(data, model)
        
        print('PRED:',pred)
		
        lst = []
        
        if pred[0] == 1:
            lst.append('will')
            lst.append('1F61F')
        else:
            lst.append('will not')
            lst.append('1F642')
            
    return render_template('result.html',prediction = lst)

#Boiler plate code
#run method starts our web server
#debug True: Restarts server based on any code change
if __name__ == '__main__':
    app.run(debug=True)
