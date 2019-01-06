# -*- coding: utf-8 -*-
"""
Created on Sun Jan  6 09:52:57 2019

@author: ratin
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Jan  6 03:08:49 2019

@author: ratin
"""

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import pandas as pd
from subprocess import check_output
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt

print("asd")
#-----------------------------------------------------------------------------





def downloader(name):
    #selector = body > div.site-layout > div.site-layout__main-content > div > div > div > div.dataset-data > div.dataset-data__content > div > div.content-box > div:nth-child(1) > div > div.content-box__right-side > div > div:nth-child(1) > div > div.StyledHintDetailed-QHccL.lcVvAz.Wrapper-bMigRJ.fSdEhz.ApiHintContainer-eyqxAC.ciYWV > span.tooltip-multiline.StyledTooltip-fWedUV.efsxir.CenteredToolTip-gSyOaz.fCfSkP > button > div

    if name.split()[0]=='kaggle': ##Implies API call
        try:
            print(check_output(name , shell=True).decode())
        except:
            print("Some unknown err occured")
    else:
        print("passed parameter should start with \"kaggle\"")


app = dash.Dash()
df = pd.read_csv("winequalityN.csv")

drop1_options = []
drop2_options = []
for column in df.columns:
    if type(df[column][1])!= str:
        drop1_options.append({'label' : str(column) , 'value': str(column)})
        drop2_options.append({'label' : str(column) , 'value': str(column)})


app.layout = html.Div([
    dcc.Graph(id='graph'),html.Div([
    dcc.Dropdown(id="tipe",options=[{'label': i, 'value': i} for i in ['LinearRegression', 'Classification']] , value = "LinearRegression")]),
    html.Div([
    html.Div([dcc.Dropdown(id="drop1", options = drop1_options , value = drop1_options[0]["value"]),
    dcc.Dropdown(id="drop2", options = drop2_options , value = drop2_options[0]["value"])]),
    dcc.RadioItems(
                id='radio',
                options=[{'label': i, 'value': i} for i in ['ShowLine', 'NotShowLine']],
                value='Linear',
                labelStyle={'display': 'inline-block'}
            )
    ])
])

@app.callback(Output('graph' , 'figure'),
              [Input('tipe','value'),Input('drop1','value'),Input('drop2', 'value'),Input('radio','value')])
def update_figure(tipe,column1,column2,radio):
    df.dropna(inplace=True)
    x = df[column1]
    y = df[column2]
    traces =[]
    traces.append(go.Scatter(
        x = x,
        y = y,
        mode= "markers",
        marker={
                'size': 15,
                'opacity': 0.5,
                'line': {'width': 0.5, 'color': 'white'}
            }
    ))
    if radio =="ShowLine":
        x1= x.values.reshape(-1,1)
        y1=y.values
#        scaler = StandardScaler().fit(x1)
#        x1 = scaler.transform(x1)
        lr = LinearRegression()
        lr.fit(x1,y1)
        plt.scatter(x1,y1)
        plt.scatter(x1,lr.predict(x1))
        plt.show()
        print(x1.shape, lr.predict(x1).shape,"reshape ",x1.reshape(-1,1).shape)
        
        traces.append(go.Scatter(
            x = x1.reshape(-1),
            y = lr.predict(x1),
            mode = 'lines'
        ))
    
    print(len(traces))

    return {'data': traces , 'layout': go.Layout(title="myPLOT",
           xaxis={
                'title': column1,
            },
            yaxis={
                'title': column2,
            })}

if __name__=="__main__":
    app.run_server()
    
    


X = np.random.random((10,5))
y = np.array(['M','M','F','F','M','F','M','M','F','F'])

X[X < 0.7] = 0
scaler = StandardScaler().fit(X)
standardized_X = scaler.transform(X)
standardized_X_test = scaler.transform(X)



enc = LabelEncoder()
y = enc.fit_transform(y)

from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

from sklearn.linear_model import LinearRegression  
regressor = LinearRegression()  
regressor.fit(X_train, y_train)
print(regressor.intercept_,regressor.coef_)
y_pred = regressor.predict(X_test)  

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv("winequalityN.csv")
dataset.dropna(inplace = True)
X = dataset.iloc[:, 2].values.reshape(-1,1)
y = dataset.iloc[:, 1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

# Fitting Simple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)

# Visualising the Training set results
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.show()

# Visualising the Test set results
plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

plt.scatter(X_train,y_train)
plt.plot(X_trae)