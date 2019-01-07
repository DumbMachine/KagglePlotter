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
    dcc.Dropdown(id="tipe",options=[{'label': i, 'value': i} for i in ['Scatter', 'Boxplots', 'Line', 'Bubble']] , value = "Scatter")]),
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
    if tipe =='Scatter':
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
        
    elif tipe=="Boxplots":
        trace0 = go.Box(
            y=x,
            name = column2
        )
        trace1 = go.Box(
            y=y,
            name = column1
        )
        traces.append(trace0)
        traces.append(trace1)
        col = column2
        if df[column1].mean()>df[column2].mean():
            col = column1
        
        return {'data': traces , 'layout': go.Layout(
                    yaxis=dict(
                        range=[df[col].min(), df[col].max()]
                    )
                )}

       
        
    elif tipe == "Bubble":
        traces.append(go.Scatter(
        x = x,
        y = y,
        mode= "markers",
        marker={
                'size': 3*df["alcohol"], #TODO: 3*df[choice_column]
                'color' : df["alcohol"],
                'opacity': 0.3,
                'showscale' : True,
                'line': {'width': 0.5, 'color': 'white'}
            }
    ))
        
    elif tipe == "Line":
        traces.append(go.Scatter(
        x = x,
        y = y,
        mode= "lines",
        line = dict(
        color = ('rgb(0, 0, 0)'),
        width = 4,
        dash = 'dash')
        )
    )
        
        
    if radio =="ShowLine":
        x1= x.values.reshape(-1,1)
        y1=y.values
#        scaler = StandardScaler().fit(x1)
#        x1 = scaler.transform(x1)
        lr = LinearRegression()
        lr.fit(x1,y1)
#        plt.scatter(x1,y1)
#        plt.scatter(x1,lr.predict(x1))
#        plt.show()
#        print(x1.shape, lr.predict(x1).shape,"reshape ",x1.reshape(-1,1).shape)
        
        traces.append(go.Scatter(
            x = x1.reshape(-1),
            y = lr.predict(x1),
            mode = 'lines',
            line = dict(
        color = ('rgb(0, 0, 0)'),
        width = 4,
        dash = 'dash')
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
    