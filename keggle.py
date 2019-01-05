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
import matplotlib.pyplot as plt
from subprocess import check_output
import requests
from bs4 import BeautifulSoup

df = pd.read_csv("winequalityN.csv")
app = dash.Dash()
#-----------------------------------------------------------------------------
drop1_options = []
drop2_options = []
for column in df.columns:
    if type(df[column][1])!= str:
        drop1_options.append({'label' : str(column) , 'value': str(column)})
        drop2_options.append({'label' : str(column) , 'value': str(column)})


app.layout = html.Div([
    dcc.Graph(id='graph'),
    html.Div([
    dcc.Dropdown(id="drop1", options = drop1_options , value = drop1_options[0]["value"]),
    dcc.Dropdown(id="drop2", options = drop2_options , value = drop2_options[0]["value"])
    ])
])

@app.callback(Output('graph' , 'figure'),
              [Input('drop1','value'),Input('drop2', 'value')])
def update_figure(column1,column2):
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

    return {'data': traces , 'layout': go.Layout(title="myPLOT",
           xaxis={
                'title': column1,
            },
            yaxis={
                'title': column2,
            })}

if __name__=="__main__":
    app.run_server()


def downloader(name):
    #selector = body > div.site-layout > div.site-layout__main-content > div > div > div > div.dataset-data > div.dataset-data__content > div > div.content-box > div:nth-child(1) > div > div.content-box__right-side > div > div:nth-child(1) > div > div.StyledHintDetailed-QHccL.lcVvAz.Wrapper-bMigRJ.fSdEhz.ApiHintContainer-eyqxAC.ciYWV > span.tooltip-multiline.StyledTooltip-fWedUV.efsxir.CenteredToolTip-gSyOaz.fCfSkP > button > div

    if name.split()[0]=='kaggle': ##Implies API call
        try:
            print(check_output("name" , shell=True))
        except:
            print("Some unknown err occured")
    else:
        print("API was wrong")

