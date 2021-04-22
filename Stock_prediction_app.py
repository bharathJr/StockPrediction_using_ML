#!/usr/bin/env python
# coding: utf-8

# In[1]:


import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import plotly.graph_objs as go
from dash.dependencies import Input, Output
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import numpy as np

import plotly.express as px
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

colors = {
    'background': '#111111',
    'text': '#7FDBFF'
}




# app = dash.Dash()

server = app.server

scaler=MinMaxScaler(feature_range=(0,1))



df_nse = pd.read_csv("./GOOG_2011_new.csv")

df_nse["Date"]=pd.to_datetime(df_nse.Date,format="%Y-%m-%d")
df_nse.index=df_nse['Date']


data=df_nse.sort_index(ascending=True,axis=0)
new_data=pd.DataFrame(index=range(0,len(df_nse)),columns=['Date','Close'])

for i in range(0,len(data)):
    new_data["Date"][i]=data['Date'][i]
    new_data["Close"][i]=data["Close"][i]

new_data.index=new_data.Date
new_data.drop("Date",axis=1,inplace=True)

dataset=new_data.values

train=dataset[0:1500,:]
valid=dataset[1500:,:]

scaler=MinMaxScaler(feature_range=(0,1))
scaled_data=scaler.fit_transform(dataset)

x_train,y_train=[],[]

for i in range(60,len(train)):
    x_train.append(scaled_data[i-60:i,0])
    y_train.append(scaled_data[i,0])
    
x_train,y_train=np.array(x_train),np.array(y_train)

x_train=np.reshape(x_train,(x_train.shape[0],x_train.shape[1],1))

model=load_model("ModelNEW2011.h5")

inputs=new_data[len(new_data)-len(valid)-60:].values
inputs=inputs.reshape(-1,1)
inputs=scaler.transform(inputs)

X_test=[]
for i in range(60,inputs.shape[0]):
    X_test.append(inputs[i-60:i,0])
X_test=np.array(X_test)

X_test=np.reshape(X_test,(X_test.shape[0],X_test.shape[1],1))
closing_price=model.predict(X_test)
closing_price=scaler.inverse_transform(closing_price)

train=new_data[1500:]
valid=new_data[1500:]
valid['Predictions']=closing_price



df= pd.read_csv("./BIG_DATA_Final.csv")

app.layout = html.Div(style={'backgroundColor': colors['background'],'color': colors['text']}, children=[
   
    html.H1("Stock Price Analysis Dashboard", style={"textAlign": "center"}),
   
    dcc.Tabs(id="tabs",children=[
        

        dcc.Tab(label='GOOGLE Stock Data',children=[
			html.Div(style={'backgroundColor': colors['background']}, children=[
				html.H2("Actual closing price", style={"textAlign": "center"}),
				dcc.Graph(
					id="Actual Data",
					figure={
						"data":[
							go.Scatter(
								x=train.index,
								y=valid["Close"],
								mode='markers'
							)

						],
						"layout":go.Layout(
							title='scatter plot',
							xaxis={'title':'Date'},
							yaxis={'title':'Closing Rate'},
              plot_bgcolor=colors['background'],
              paper_bgcolor=colors['background'],
              font_color=colors['text']
						)
					}

				),
				html.H2("LSTM Predicted closing price", style={"textAlign": "center"}),
				dcc.Graph(
					id="Predicted Data",
					figure={
						"data":[
							go.Scatter(
								x=valid.index,
								y=valid["Predictions"],
								mode='markers'
							)

						],
						"layout":go.Layout(
							title='scatter plot',
							xaxis={'title':'Date'},
							yaxis={'title':'Closing Rate'},
              plot_bgcolor=colors['background'],
              paper_bgcolor=colors['background'],
              font_color=colors['text']
						)
					}

				)				
			])        		


        ]),
        dcc.Tab(label='Tech Companies Stock Data', children=[
            html.Div([
                html.H1("High and Low Stocks Price comparison(USD)", 
                        style={'textAlign': 'center'}),
              
                dcc.Dropdown(id='my-dropdown',
                             options=[{'label': 'Tesla', 'value': 'TSLA'},
                                      {'label': 'Apple','value': 'AAPL'}, 
                                      {'label': 'Facebook', 'value': 'FB'}, 
                                      {'label': 'Microsoft','value': 'MSFT'}, 
                                      {'label': 'GOOGLE','value': 'GOOG'},
                                      {'label': 'Amazon','value': 'AMZN'}, 
                                      {'label': 'IBM','value': 'IBM'},
                                      {'label': 'NETFLIX','value': 'NFLX'}, 
                                      {'label': 'Tata Motors','value': 'TTM'},
                                      {'label': 'Adobe','value': 'ADBE'},
                                      {'label': 'Oracle','value': 'ORCL'},
                                      {'label': 'Accenture','value': 'ACN'},
                                      {'label': 'AMD','value': 'AMD'},
                                      {'label': 'NVIDIA','value': 'NVDA'},
                                      {'label': 'Intel','value': 'INTC'}], 
                             multi=True,value=['TSLA'],
                             style={"display": "block", "margin-left": "auto", 
                                    "margin-right": "auto", "width": "80%",
                                    'backgroundColor': colors['background']}),
                dcc.Graph(id='highlow'),
                html.H1("Stocks Market Volume", style={'textAlign': 'center'}),
         
                dcc.Dropdown(id='my-dropdown2',
                             options=[{'label': 'Tesla', 'value': 'TSLA'},
                                      {'label': 'Apple','value': 'AAPL'}, 
                                      {'label': 'Facebook', 'value': 'FB'}, 
                                      {'label': 'Microsoft','value': 'MSFT'}, 
                                      {'label': 'GOOGLE','value': 'GOOG'},
                                      {'label': 'Amazon','value': 'AMZN'}, 
                                      {'label': 'IBM','value': 'IBM'},
                                      {'label': 'NETFLIX','value': 'NFLX'}, 
                                      {'label': 'Tata Motors','value': 'TTM'},
                                      {'label': 'Adobe','value': 'ADBE'},
                                      {'label': 'Oracle','value': 'ORCL'},
                                      {'label': 'Accenture','value': 'ACN'},
                                      {'label': 'AMD','value': 'AMD'},
                                      {'label': 'NVIDIA','value': 'NVDA'},
                                      {'label': 'Intel','value': 'INTC'}], 
                             multi=True,value=['TSLA'],
                             style={"display": "block", "margin-left": "auto", 
                                    "margin-right": "auto", "width": "80%",
                                    'backgroundColor': colors['background']}),
                dcc.Graph(id='volume')
            ], className="container"),
        ])


    ])
])







@app.callback(Output('highlow', 'figure'),
              [Input('my-dropdown', 'value')])
def update_graph(selected_dropdown):
    dropdown = {"TSLA": "Tesla","AAPL": "Apple","FB": "Facebook","MSFT": "Microsoft","AMZN": "AMAZON","GOOG": "GOOGLE",
                "IBM": "IBM","NFLX": "NETFLIX","TTM": "Tata Motors","ADBE": "Adobe","ORCL": "Oracle","ACN": "Accenture",
                "AMD": "AMD","NVDA": "NVIDIA","INTC": "Intel", }
    trace1 = []
    trace2 = []
    for stock in selected_dropdown:
        trace1.append(
          go.Scatter(x=df[df["Stock"] == stock]["Date"],
                     y=df[df["Stock"] == stock]["High"],
                     mode='lines', opacity=0.7, 
                     name=f'High {dropdown[stock]}',textposition='bottom center'))
        trace2.append(
          go.Scatter(x=df[df["Stock"] == stock]["Date"],
                     y=df[df["Stock"] == stock]["Low"],
                     mode='lines', opacity=0.6,
                     name=f'Low {dropdown[stock]}',textposition='bottom center'))
    traces = [trace1, trace2]
    data = [val for sublist in traces for val in sublist]
    figure = {'data': data,
              'layout': go.Layout(colorway=["#5E0DAC", '#FF4F00', '#375CB1', 
                                            '#FF7400', '#FFF400', '#FF0056',
                                            '#0D0F75', '#0A0EF6', '#FF041F',
                                            '#06F51F', '#D9F90C', '#FF8000',
                                            '#FF4200', '#04F7E5', '#04B1F7'],
            height=600,
            title=f"High and Low Prices for {', '.join(str(dropdown[i]) for i in selected_dropdown)} Over Time",
            xaxis={"title":"Date",
                   'rangeselector': {'buttons': list([{'count': 1, 'label': '1M', 
                                                       'step': 'month', 
                                                       'stepmode': 'backward'},
                                                      {'count': 6, 'label': '6M', 
                                                       'step': 'month', 
                                                       'stepmode': 'backward'},
                                                      {'step': 'all'}])},
                   'rangeslider': {'visible': True}, 'type': 'date'},
             yaxis={"title":"Price (USD)"},
             plot_bgcolor=colors['background'],
             paper_bgcolor=colors['background'])}
    return figure


@app.callback(Output('volume', 'figure'),
              [Input('my-dropdown2', 'value')])
def update_graph(selected_dropdown_value):
    dropdown = {"TSLA": "Tesla","AAPL": "Apple","FB": "Facebook","MSFT": "Microsoft","AMZN": "AMAZON","GOOG": "GOOGLE",
                "IBM": "IBM","NFLX": "NETFLIX","TTM": "Tata Motors","ADBE": "Adobe","ORCL": "Oracle","ACN": "Accenture",
                "AMD": "AMD","NVDA": "NVIDIA","INTC": "Intel", }
    trace1 = []
    for stock in selected_dropdown_value:
        trace1.append(
          go.Scatter(x=df[df["Stock"] == stock]["Date"],
                     y=df[df["Stock"] == stock]["Volume"],
                     mode='lines', opacity=0.7,
                     name=f'Volume {dropdown[stock]}', textposition='bottom center'))
    traces = [trace1]
    data = [val for sublist in traces for val in sublist]
    figure = {'data': data, 
              'layout': go.Layout(colorway=["#5E0DAC", '#FF4F00', '#375CB1', 
                                            '#FF7400', '#FFF400', '#FF0056',
                                            '#0D0F75', '#0A0EF6', '#FF041F',
                                            '#06F51F', '#D9F90C', '#FF8000',
                                            '#FF4200', '#04F7E5', '#04B1F7'],
            height=600,
            title=f"Market Volume for {', '.join(str(dropdown[i]) for i in selected_dropdown_value)} Over Time",
            xaxis={"title":"Date",
                   'rangeselector': {'buttons': list([{'count': 1, 'label': '1M', 
                                                       'step': 'month', 
                                                       'stepmode': 'backward'},
                                                      {'count': 6, 'label': '6M',
                                                       'step': 'month', 
                                                       'stepmode': 'backward'},
                                                      {'step': 'all'}])},
                   'rangeslider': {'visible': True}, 'type': 'date'},
             yaxis={"title":"Transactions Volume"},
             plot_bgcolor=colors['background'],
             paper_bgcolor=colors['background'])}
    return figure



if __name__=='__main__':
	app.run_server(debug=True)


# In[ ]:




