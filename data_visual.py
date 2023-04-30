import yfinance as yf
import numpy as np
import pandas as pd
import mpld3
from numpy import array
import pickle
import math
import xlrd

import seaborn as sns
import matplotlib.pyplot as plt
import plotly.figure_factory as ff 
import plotly.graph_objects as go
import plotly.express as px
import plotly
import plotly.subplots as sp
import plotly.io as pio
from sklearn.preprocessing import MinMaxScaler 

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import pacf
from statsmodels.tsa.stattools import acf

#import tensorflow as tf
#from tensorflow import keras
#from keras import layers
#from keras.models import Sequential
#from keras.layers import LSTM
#from keras.layers import Dense 

#Display table
def display_data_price():
    gas = pd.read_excel('NaturalGas.xls', sheet_name="Data 1", skiprows=2)
    gas['Date']=pd.to_datetime(gas['Date'])
    gas.index= pd.DatetimeIndex(gas['Date'])
    df1=gas[['Date', 'U.S. Natural Gas Wellhead Price (Dollars per Thousand Cubic Feet)']].dropna().head(5)
    df1 = df1[df1.iloc[:,1] != 0]
    df = ff.create_table(df1,height_constant=20)
    return plotly.offline.plot(df,output_type='div')

def display_data_prod():
    gas = pd.read_excel('NaturalGas.xls', sheet_name="Data 2", skiprows=2)
    gas['Date']=pd.to_datetime(gas['Date'])
    gas.index= pd.DatetimeIndex(gas['Date'])
    df1=gas[['Date', 'U.S. Natural Gas Gross Withdrawals (MMcf)']].dropna().head(5)
    df1 = df1[df1.iloc[:,1] != 0]
    df = ff.create_table(df1,height_constant=20)
    return plotly.offline.plot(df,output_type='div')

def display_data_import_exports():
    gas = pd.read_excel('NaturalGas.xls', sheet_name="Data 3", skiprows=2)
    gas['Date']=pd.to_datetime(gas['Date'])
    gas.index= pd.DatetimeIndex(gas['Date'])
    df1=gas[['Date', 'U.S. Natural Gas Imports (MMcf)', 'U.S. Natural Gas Exports (MMcf)']].dropna().head(5)
    df1 = df1[df1.iloc[:,1] != 0]
    df1 = df1[df1.iloc[:,2] != 0]
    df = ff.create_table(df1,height_constant=20)
    return plotly.offline.plot(df,output_type='div')

def display_data_consumptions():
    gas = pd.read_excel('NaturalGas.xls', sheet_name="Data 5", skiprows=2)
    gas['Date'] = pd.to_datetime(gas['Date'])
    gas.index= pd.DatetimeIndex(gas['Date'])
    gas.drop(['U.S. Natural Gas Total Consumption (MMcf)'], axis = 1)
    df1 = gas.dropna().head(5)
    for i in range(df1.shape[1]):
        df1 = df1[df1.iloc[:,i] != 0]

    #df1.rename(columns={'U.S. Natural Gas Lease and Plant Fuel Consumption (MMcf)':'Gas and plant'})
    df = ff.create_table(df1,height_constant=20)
    df.layout.width = 4000
    return plotly.offline.plot(df,output_type='div')

def input_price():
    gas = pd.read_excel('NaturalGas.xls', sheet_name="Data 1", skiprows=2)
    gas['Date']=pd.to_datetime(gas['Date'])
    gas.index= pd.DatetimeIndex(gas['Date'])
    gas['month']=[gas.index[i].month for i in range(len(gas))]
    gas['year']=[gas.index[i].year for i in range(len(gas))]
    df1=gas[['U.S. Natural Gas Wellhead Price (Dollars per Thousand Cubic Feet)', 'Date']].copy().dropna()
    return df1

def input_prod():
    gas = pd.read_excel('NaturalGas.xls', sheet_name="Data 2", skiprows=2)
    gas['Date']=pd.to_datetime(gas['Date'])
    gas.index= pd.DatetimeIndex(gas['Date'])
    gas['month']=[gas.index[i].month for i in range(len(gas))]
    gas['year']=[gas.index[i].year for i in range(len(gas))]
    df1=gas[['U.S. Natural Gas Gross Withdrawals (MMcf)', 'Date']].copy().dropna()
    return df1

def plot1():
    df1=input_price()
    fig = px.line(df1, x='Date', y='U.S. Natural Gas Wellhead Price (Dollars per Thousand Cubic Feet)')
    fig.update_xaxes(rangeslider_visible=True)
    fig.update_layout(
    title='Time Series with Rangeslider for Wellhead Price - Univariate TS PLOT',
    xaxis_title='Date',
    yaxis_title='Price')
    pickle.dump(plotly.offline.plot(fig, output_type='div'),open("plot1.pkl","wb"))
    #return plotly.offline.plot(fig, output_type='div')

def input_gas():
    gas = pd.read_excel('NaturalGas.xls', sheet_name="Data 1", skiprows=2)
    gas['Date']=pd.to_datetime(gas['Date'])
    gas.index= pd.DatetimeIndex(gas['Date'])
    gas['month']=[gas.index[i].month for i in range(len(gas))]
    gas['year']=[gas.index[i].year for i in range(len(gas))]
    return gas

def uni_plot2():
    gas=input_gas()
    fig = px.line(gas, x='month', y='U.S. Natural Gas Wellhead Price (Dollars per Thousand Cubic Feet)', color='year',
              title='Seasonal plot')
    fig.update_layout(legend=dict(title='Year', yanchor='top', y=0.99, xanchor='left', x=0.01))
    pickle.dump(plotly.offline.plot(fig,output_type='div'), open("uni_plot2.pkl","wb"))
    #return plotly.offline.plot(fig,output_type='div')

def uni_plot3():
    df=input_gas()
    df=df[['U.S. Natural Gas Wellhead Price (Dollars per Thousand Cubic Feet)']].copy().dropna()
    res1 = seasonal_decompose(df, model='multiplicative', period=80)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=res1.observed.index, y=res1.observed, name='Observed'))
    fig.add_trace(go.Scatter(x=res1.trend.index, y=res1.trend, name='Trend'))
    fig.add_trace(go.Scatter(x=res1.seasonal.index, y=res1.seasonal, name='Seasonal'))
    fig.add_trace(go.Scatter(x=res1.resid.index, y=res1.resid, name='Residual'))
    fig.update_layout(
            title="Multiplicative Decomposition",
            xaxis_title="Date",
            yaxis_title="Wellhead Price Value"
        )
    pickle.dump(plotly.offline.plot(fig,output_type='div'), open("uni_plot3.pkl","wb"))
    #return plotly.offline.plot(fig,output_type='div')

def auto_plot1():
    gas = pd.read_excel('NaturalGas.xls', sheet_name="Data 1", skiprows=2)
    gas['Date']=pd.to_datetime(gas['Date'])
    gas.index= pd.DatetimeIndex(gas['Date'])
    df1=gas[['U.S. Natural Gas Wellhead Price (Dollars per Thousand Cubic Feet)']].copy().dropna()
    df_acf = acf(df1, nlags=300)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
            x= np.arange(len(df_acf)),
            y= df_acf,
            name= 'ACF',
            ))
    fig.update_xaxes(rangeslider_visible=True)
    fig.update_layout(
            title="Autocorrelation",
            xaxis_title="Lag",
            yaxis_title="Autocorrelation",
            #     autosize=False,
            #     width=500,
                height=500,
            )
    pickle.dump(plotly.offline.plot(fig,output_type='div'), open("auto_plot1.pkl","wb"))
    #return plotly.offline.plot(fig,output_type='div')

def auto_plot2():
    gas = pd.read_excel('NaturalGas.xls', sheet_name="Data 1", skiprows=2)
    gas['Date']=pd.to_datetime(gas['Date'])
    gas.index= pd.DatetimeIndex(gas['Date'])
    df1=gas[['U.S. Natural Gas Wellhead Price (Dollars per Thousand Cubic Feet)']].copy().dropna()
    df_pacf = pacf(df1, nlags=220)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x= np.arange(len(df_pacf)),
        y= df_pacf,
        name= 'PACF',
        ))
    fig.update_xaxes(rangeslider_visible=True)
    fig.update_layout(
        title="Partial Autocorrelation",
        xaxis_title="Lag",
        yaxis_title="Partial Autocorrelation",
        #     autosize=False,
        #     width=500,
            height=500,
        )
    pickle.dump(plotly.offline.plot(fig,output_type='div'), open("auto_plot2.pkl","wb"))
    #return plotly.offline.plot(fig,output_type='div')

def input3():
    gas = pd.read_excel('NaturalGas.xls', sheet_name="Data 1", skiprows=2)
    gas['Date']=pd.to_datetime(gas['Date'])
    gas.index= pd.DatetimeIndex(gas['Date'])
    gas['month']=[gas.index[i].month for i in range(len(gas))]
    gas['year']=[gas.index[i].year for i in range(len(gas))]
    return gas

def input2():
    gas = pd.read_excel('NaturalGas.xls', sheet_name="Data 3", skiprows=2)
    gas['Date']=pd.to_datetime(gas['Date'])
    gas.index= pd.DatetimeIndex(gas['Date'])
    gas['month']=[gas.index[i].month for i in range(len(gas))]
    gas['year']=[gas.index[i].year for i in range(len(gas))]
    df1=gas[['U.S. Natural Gas Imports (MMcf)', 'Date', 'U.S. Natural Gas Exports (MMcf)']].copy().dropna()
    return df1

def plot2():
    df1=input2()
    fig = px.scatter(df1, x="U.S. Natural Gas Imports (MMcf)", y="U.S. Natural Gas Exports (MMcf)", 
                   color_continuous_scale=px.colors.sequential.Agsunset, render_mode="webgl",title="Scatter Plot Export Vs Imports for Natural Gas - Multivariate TS PLOT")
    pickle.dump(plotly.offline.plot(fig,output_type='div'), open("plot2.pkl","wb"))
    #return plotly.offline.plot(fig,output_type='div')

#Forecasting plots- ARIMA

def forecast_input():
    gas = pd.read_excel('NaturalGas.xls', sheet_name="Data 1", skiprows=2)
    gas['Date']=pd.to_datetime(gas['Date'])
    gas.index= pd.DatetimeIndex(gas['Date'])
    gas['month']=[gas.index[i].month for i in range(len(gas))]
    gas['year']=[gas.index[i].year for i in range(len(gas))]
    df1=gas[['U.S. Natural Gas Wellhead Price (Dollars per Thousand Cubic Feet)']].copy().dropna()
    return df1


df1=forecast_input()
df1=df1.diff()
model = ARIMA(df1, order=(4,3,1))
model_fit = model.fit()
pickle.dump(model_fit, open("price_arima.pkl", "wb"))  

#Forecasting plots- VAR

def forecast_input2():
    gas = pd.read_excel('NaturalGas.xls', sheet_name="Data 3", skiprows=2)
    gas['Date']=pd.to_datetime(gas['Date'])
    gas.index= pd.DatetimeIndex(gas['Date'])

    df2=gas[['U.S. Natural Gas Imports (MMcf)','U.S. Natural Gas Exports (MMcf)']].copy()
    df2=df2.dropna()

    df_diff = df2.diff().dropna()
        
    import_export_df = df_diff[['U.S. Natural Gas Imports (MMcf)', 'U.S. Natural Gas Exports (MMcf)']]

    return import_export_df

def var_impexp_fit():
    df_var=forecast_input2()
    model_var = VAR(df_var[:-12])
    model_fit_var = model_var.fit(maxlags = 13)
    pickle.dump(model_fit_var,open("imp-exp_var.pkl","wb"))

def input1():
    gas = pd.read_excel('NaturalGas.xls', sheet_name="Data 3", skiprows=2)
    gas['Date']=pd.to_datetime(gas['Date'])
    gas.index= pd.DatetimeIndex(gas['Date'])

    df2=gas[['U.S. Natural Gas Imports (MMcf)','U.S. Natural Gas Exports (MMcf)']].copy()
    df2=df2.dropna()

    df_diff = df2.diff().dropna()

    training_set = df_diff[:int(0.95*(len(df2)))]
    test_set = df_diff[int(0.95*(len(df2))):]
        
    import_export_df = df_diff[['U.S. Natural Gas Imports (MMcf)', 'U.S. Natural Gas Exports (MMcf)']]
    return test_set,import_export_df,df_diff

def forecast_var_impexp():
    model3=pickle.load(open("imp-exp_var.pkl","rb"))
    test_set,import_export_df,df_diff=input1()
    pred = model3.forecast(test_set.values, steps=len(test_set))
    aic_var = model3.aic
    bic_var = model3.bic
    forecast = pd.DataFrame(pred, index=test_set.index, columns=test_set.columns)
    #rmse = sqrt(mean_squared_error(test_set, forecast))
    # Create traces for actual and predicted values
    actual_trace = go.Scatter(x=test_set.index, y=test_set['U.S. Natural Gas Imports (MMcf)'],
                                mode='lines', name='Actual')
    predicted_trace = go.Scatter(x=forecast.index, y=forecast['U.S. Natural Gas Imports (MMcf)'],
                                mode='lines', name='Predicted')

    # Create the figure and add traces
    fig1 = go.Figure()
    fig1.add_trace(actual_trace)
    fig1.add_trace(predicted_trace)

    # Add title and axis labels
    fig1.update_layout(title='Actual vs Predicted U.S. Natural Gas Imports (MMcf)',
                        xaxis_title='Date', yaxis_title='U.S. Natural Gas Imports (MMcf)')

  
    plot_html1 = fig1.to_html(full_html=False)
    pickle.dump(plot_html1, open("var_impexp_fig1.pkl","wb"))
 
    forecast_df = pd.DataFrame(forecast, index=test_set.index, columns=test_set.columns)

  # Invert the differenced values to get the actual gas imports and exports data
    forecast_df['U.S. Natural Gas Imports (MMcf)'] = forecast_df['U.S. Natural Gas Imports (MMcf)'].cumsum() + import_export_df['U.S. Natural Gas Imports (MMcf)'].iloc[-121]
    forecast_df['U.S. Natural Gas Exports (MMcf)'] = forecast_df['U.S. Natural Gas Exports (MMcf)'].cumsum() + import_export_df['U.S. Natural Gas Exports (MMcf)'].iloc[-121]

    # Plot the forecasted values
    import plotly.express as px

    fig2 = px.line(df_diff, y=['U.S. Natural Gas Imports (MMcf)', 'U.S. Natural Gas Exports (MMcf)'], 
                    title='Historical Imports and Exports')

    fig2.add_scatter(x=forecast_df.index, y=forecast_df['U.S. Natural Gas Imports (MMcf)'], mode='lines', 
                    name='Forecasted Imports')

    fig2.add_scatter(x=forecast_df.index, y=forecast_df['U.S. Natural Gas Exports (MMcf)'], mode='lines', 
                    name='Forecasted Exports')

    plot_html2 = fig2.to_html(full_html=False)
    pickle.dump(plot_html2, open("var_impexp_fig2.pkl","wb"))

#NATURAL GAS- PRODUCTION

def prod_input():
    data = pd.read_excel('NaturalGas.xls', sheet_name = "Data 2", skiprows = 2)
    data = data.dropna()
    data['Date'] = pd.to_datetime(data['Date'])
    data.index = pd.DatetimeIndex(data['Date'])
    df = data[['U.S. Natural Gas Gross Withdrawals (MMcf)']].copy()
    return df

def prod_plot3():
    df=prod_input()
    # Create a box plot
    fig = px.box(df, x=df.index.month, y='U.S. Natural Gas Gross Withdrawals (MMcf)', title='Natural Gas Gross Withdrawals by Month')
    fig.update_xaxes(title='Months')
    pickle.dump(plotly.offline.plot(fig,output_type='div'), open("prod_plot3.pkl","wb"))
    #return plotly.offline.plot(fig,output_type='div')

def prod_plot4():
    df=prod_input()
    df_yoy = df.groupby(df.index.year).sum().pct_change()
    fig = px.bar(df_yoy, x=df_yoy.index, y='U.S. Natural Gas Gross Withdrawals (MMcf)', title='Year-over-Year Change in Natural Gas Gross Withdrawals')
    pickle.dump(plotly.offline.plot(fig,output_type='div'), open("prod_plot4.pkl","wb"))
    #return plotly.offline.plot(fig,output_type='div')

#Lag Pot
def prod_plot1():
    df=prod_input()
    df_shift = pd.concat([df, df.shift()], axis=1)
    df_shift.columns = ["y", "y_lag1"]
    df_shift = df_shift.dropna()
    fig = px.scatter(df_shift, x="y_lag1", y="y", trendline="ols")
    fig.update_layout(
    title={
        'text': "Lag Plot of Natural Gas Production",
        'y':0.95,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'
    }
    )
    pickle.dump(plotly.offline.plot(fig,output_type='div'), open("prod_plot1.pkl","wb"))
    #return plotly.offline.plot(fig,output_type='div')

#additive decomposition plot
def prod_plot2():
    df=prod_input()
    res1 = seasonal_decompose(df, model='additive', period=40)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=res1.observed.index, y=res1.observed, name='Observed'))
    fig.add_trace(go.Scatter(x=res1.trend.index, y=res1.trend, name='Trend'))
    fig.add_trace(go.Scatter(x=res1.seasonal.index, y=res1.seasonal, name='Seasonal'))
    fig.add_trace(go.Scatter(x=res1.resid.index, y=res1.resid, name='Residual'))
    fig.update_layout(
        title="Additive Decomposition",
        xaxis_title="Date",
        yaxis_title="Production Value"
    )
    pickle.dump(plotly.offline.plot(fig,output_type='div'), open("prod_plot2.pkl","wb"))
    #return plotly.offline.plot(fig,output_type='div')


#Consumption Visualizations

def con_input():
    data = pd.read_excel('NaturalGas.xls', sheet_name = "Data 5", skiprows = 2)
    #drop na values 
    data = data.dropna()
    data.index = pd.DatetimeIndex(data['Date'])
    #Convert the Date column into a date object
    data['Date'] = pd.to_datetime(data['Date'])
    df = data.drop(['Date', 'U.S. Natural Gas Total Consumption (MMcf)'] , axis = 1 )
    return df

def con_plot1():
    df=con_input()
    fig = px.line(df, x=df.index, y=df.columns, title='Consumption of Natural Gas over Different Sources')
    fig.update_xaxes(title='Date')
    fig.update_yaxes(title='Values')
    pickle.dump(plotly.offline.plot(fig,output_type='div'), open("con_plot1.pkl","wb"))
    #return plotly.offline.plot(fig,output_type='div')

def con_plot2():
    df=con_input()
    fig = px.box(data_frame=df, x=df.index.year, y=df.columns,
             title='Natural Gas Consumption by Year', color=df.index.year)
    fig.update_xaxes(title='Year')
    fig.update_yaxes(title='Consumption of Natural Gas over different sources')
    pickle.dump(plotly.offline.plot(fig,output_type='div'), open("con_plot2.pkl","wb"))
    #return plotly.offline.plot(fig,output_type='div')

#ARIMA MODEL- Production

#VAR MODEL - Consumption
def forecast_input_var_con():
    data = pd.read_excel('NaturalGas.xls', sheet_name = "Data 5", skiprows = 2)
    #drop na values 
    data = data.dropna()
    data.index = pd.DatetimeIndex(data['Date'])
    #Convert the Date column into a date object
    data['Date'] = pd.to_datetime(data['Date'])
    df = data.drop(['Date', 'U.S. Natural Gas Total Consumption (MMcf)'] , axis = 1 )
    df_diff = df.diff().dropna()

    training_set = df_diff[:int(0.90*(len(df_diff)))]
    test_set = df_diff[int(0.90*(len(df_diff))):]
    return test_set, df_diff, data,df
    
def var_con_fit():
    df_var_con=forecast_input_var_con()
    training_set = df_var_con[:int(0.90*(len(df_var_con)))]
    test_set = df_var_con[int(0.90*(len(df_var_con))):]
        
    #Fit to a VAR model
    model_var = VAR(endog=training_set)
    #lags = model.select_order(maxlags=2)['aic']
    model_fit_var_con = model_var.fit()
    pickle.dump(model_fit_var_con,open("consumption_var.pkl","wb"))


def forecast_var_con():
    model5=pickle.load(open("consumption_var.pkl","rb"))
    test_set, df_diff, data, df = forecast_input_var_con()
    pred = model5.forecast(test_set.values, steps = 27)
    forecast = pd.DataFrame(pred, index=test_set.index, columns=test_set.columns)
    import plotly.express as px
    fig = px.line()
    fig.add_scatter(x=test_set.index, y=test_set.iloc[:,1], name='actual')
    fig.add_scatter(x=test_set.index, y=forecast.iloc[:,1], name='forecast')
    plot_html= pio.to_html(fig, full_html=False)

    pickle.dump(pio.to_html(fig, full_html=False),open("var_con_fig1.pkl","wb")) 

    n_steps = 36
    pred1 = model5.forecast(df_diff.values[-model5.k_ar:], steps=n_steps)
    forecast = pd.DataFrame(pred1, index=pd.date_range(start=data.index[-1], periods=n_steps+1, freq='MS')[1:], columns=df.columns)

    # Combine the actual and forecasted data into a single DataFrame
    combined_df = pd.concat([df_diff, forecast], axis=0)

    # Plot the actual and forecasted data using Plotly Express
    fig1 = px.line(combined_df, x=combined_df.index, y=combined_df.columns, labels={'value': 'Compumption'})
    fig1.update_layout(title='Natural Gas Consumption Forecast for the next 3 years', xaxis_title='Date')
    fig1.update_traces(mode='lines')
    plot_html1= pio.to_html(fig1, full_html=False)

    pickle.dump(pio.to_html(fig1, full_html=False),open("var_con_fig2.pkl","wb")) 

#LSTM model Prices

def input_LSTM_price():
  gas = pd.read_excel('NaturalGas.xls', sheet_name="Data 1", skiprows=2)
  gas['Date']=pd.to_datetime(gas['Date'])
  gas.index= pd.DatetimeIndex(gas['Date'])
  df1=gas[['U.S. Natural Gas Wellhead Price (Dollars per Thousand Cubic Feet)', 'Date']].copy().dropna()
  return df1

def lstm_price():

    df=input_price()
    diff = df.diff().dropna()
    naturalgas_price_import = diff['U.S. Natural Gas Wellhead Price (Dollars per Thousand Cubic Feet)']
    values = naturalgas_price_import.values
    training_data_len = math.ceil(len(values)* 0.8)

    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(values.reshape(-1,1))
    train_data = scaled_data[0: training_data_len, :]

    x_train = []
    y_train = []

    for i in range(16, len(train_data)):
        x_train.append(train_data[i-16:i, 0])
        y_train.append(train_data[i, 0])
        
    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    test_data = scaled_data[training_data_len-16: , : ]
    x_test = []
    y_test = values[training_data_len:]

    for i in range(16, len(test_data)):
        x_test.append(test_data[i-16:i, 0])

    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))


    model6 = keras.Sequential()
    model6.add(layers.LSTM(100, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model6.add(layers.LSTM(100, return_sequences=False))
    model6.add(layers.Dense(25))
    model6.add(layers.Dense(1))
    model6.summary()

    model6.compile(optimizer='adam', loss='mean_squared_error')
    model6.fit(x_train, y_train, batch_size= 1, epochs=15)
    pickle.dump(model6,open("prices_lstm.pkl","wb")) 

def forecast_lstm_price():
    model6=pickle.load(open("prices_lstm.pkl","rb"))
    df=input_LSTM_price()
    diff = df.diff().dropna()
    naturalgas_price_import = diff['U.S. Natural Gas Wellhead Price (Dollars per Thousand Cubic Feet)']
    values = naturalgas_price_import.values
    training_data_len = math.ceil(len(values)* 0.8)

    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(values.reshape(-1,1))
    train_data = scaled_data[0: training_data_len, :]

    x_train = []
    y_train = []

    for i in range(16, len(train_data)):
        x_train.append(train_data[i-16:i, 0])
        y_train.append(train_data[i, 0])
        
    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    test_data = scaled_data[training_data_len-16: , : ]
    x_test = []
    y_test = values[training_data_len:]

    for i in range(16, len(test_data)):
        x_test.append(test_data[i-16:i, 0])

    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    predictions = model6.predict(x_test)
    predictions = scaler.inverse_transform(predictions)

    data1 = diff.filter(['U.S. Natural Gas Wellhead Price (Dollars per Thousand Cubic Feet)'])
    train = data1[:training_data_len]
    validation = data1[training_data_len:]
    validation['Predictions'] = predictions


    train_trace = go.Scatter(
        x=train.index,
        y=train['U.S. Natural Gas Wellhead Price (Dollars per Thousand Cubic Feet)'],
        name='Train'
    )


    validation_trace = go.Scatter(
        x=validation.index,
        y=validation['U.S. Natural Gas Wellhead Price (Dollars per Thousand Cubic Feet)'],
        name='Validation'
    )

 
    predictions_trace = go.Scatter(
        x=validation.index,
        y=validation['Predictions'],
        name='Predictions'
    )


    fig6 = go.Figure()
    fig6.add_trace(train_trace)
    fig6.add_trace(validation_trace)
    fig6.add_trace(predictions_trace)

  
    fig6.update_layout(
        title='LSTM Model for Wellhead price of Natural Gas',
        xaxis_title='Date',
        yaxis_title='U.S. Natural Gas Wellhead Price',
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='right',
            x=1
        )
    )

    plot_html = fig6.to_html(full_html=False)
    pickle.dump(plot_html,open("forecast_lstm_price_fig1.pkl","wb"))

    n_months = 60 #5 months prediction
    last_month = x_test[-1]
    predicted_prod = []
    for i in range(n_months):
        next_month = model6.predict(last_month.reshape(1, -1, 1))
        predicted_prod.append(next_month[0, 0])
        last_month = np.append(last_month[1:], next_month)

    predicted_prod = scaler.inverse_transform(np.array(predicted_prod).reshape(-1, 1))
    last_month_date = diff.index[-1] + pd.offsets.MonthEnd()
    predicted_dates = pd.date_range(start=last_month_date, periods=n_months+1, freq='M')[1:]
    predicted_df = pd.DataFrame({'U.S. Natural Gas Wellhead Price (Dollars per Thousand Cubic Feet)': predicted_prod.flatten()}, index=predicted_dates)

    df_pred = pd.concat([diff, predicted_df])

    fig7 = go.Figure()

    fig7.add_trace(go.Scatter(x=df_pred.index, y=df_pred['U.S. Natural Gas Wellhead Price (Dollars per Thousand Cubic Feet)'],
                        mode='lines', name='Actual'))

    fig7.add_trace(go.Scatter(x=predicted_df.index, y=predicted_df['U.S. Natural Gas Wellhead Price (Dollars per Thousand Cubic Feet)'],
                        mode='lines', name='Predicted'))

    fig7.update_layout(title='Forecasting of U.S. Natural Gas Wellhead Price (Dollars per Thousand Cubic Feet)', xaxis_title='Date', yaxis_title='Prices')

    plot_html7 = fig7.to_html(full_html=False)
    pickle.dump(plot_html7,open("forecast_lstm_price_fig2.pkl","wb"))


# LSTM model production

def input_LSTM_prod():
  gas = pd.read_excel('NaturalGas.xls', sheet_name="Data 2", skiprows=2)
  gas['Date']=pd.to_datetime(gas['Date'])
  gas.index= pd.DatetimeIndex(gas['Date'])
  df1=gas[['U.S. Natural Gas Gross Withdrawals (MMcf)', 'Date']].copy().dropna()
  return df1

def lstm_prod():
    df=input_prod()
    diff = df.diff().dropna()
    naturalgas_price_import = diff['U.S. Natural Gas Gross Withdrawals (MMcf)']
    values = naturalgas_price_import.values
    training_data_len = math.ceil(len(values)* 0.8)

    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(values.reshape(-1,1))
    train_data = scaled_data[0: training_data_len, :]

    x_train = []
    y_train = []

    for i in range(16, len(train_data)):
        x_train.append(train_data[i-16:i, 0])
        y_train.append(train_data[i, 0])
        
    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    test_data = scaled_data[training_data_len-16: , : ]
    x_test = []
    y_test = values[training_data_len:]

    for i in range(16, len(test_data)):
        x_test.append(test_data[i-16:i, 0])

    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))


    model7 = keras.Sequential()
    model7.add(layers.LSTM(100, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model7.add(layers.LSTM(100, return_sequences=False))
    model7.add(layers.Dense(25))
    model7.add(layers.Dense(1))
    model7.summary()

    model7.compile(optimizer='adam', loss='mean_squared_error')
    model7.fit(x_train, y_train, batch_size= 1, epochs=15)
    pickle.dump(model7,open("prod_lstm.pkl","wb"))

def forecast_lstm_prod():
    model7=pickle.load(open("prod_lstm.pkl","rb"))
    df = input_LSTM_prod()
    diff = df.diff().dropna()
    naturalgas_prod = diff['U.S. Natural Gas Gross Withdrawals (MMcf)']
    values = naturalgas_prod.values
    training_data_len = math.ceil(len(values)* 0.8)

    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(values.reshape(-1,1))
    train_data = scaled_data[0: training_data_len, :]

    x_train = []
    y_train = []

    for i in range(16, len(train_data)):
        x_train.append(train_data[i-16:i, 0])
        y_train.append(train_data[i, 0])
        
    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    test_data = scaled_data[training_data_len-16: , : ]
    x_test = []
    y_test = values[training_data_len:]

    for i in range(16, len(test_data)):
        x_test.append(test_data[i-16:i, 0])

    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    predictions = model7.predict(x_test)
    predictions = scaler.inverse_transform(predictions)

    data1 = diff.filter(['U.S. Natural Gas Gross Withdrawals (MMcf)'])
    train = data1[:training_data_len]
    validation = data1[training_data_len:]
    validation['Predictions'] = predictions

    train_trace = go.Scatter(
        x=train.index,
        y=train['U.S. Natural Gas Gross Withdrawals (MMcf)'],
        name='Train'
    )

    validation_trace = go.Scatter(
        x=validation.index,
        y=validation['U.S. Natural Gas Gross Withdrawals (MMcf)'],
        name='Validation'
    )

 
    predictions_trace = go.Scatter(
        x=validation.index,
        y=validation['Predictions'],
        name='Predictions'
    )

    fig6 = go.Figure()

    fig6.add_trace(train_trace)
    fig6.add_trace(validation_trace)
    fig6.add_trace(predictions_trace)

    fig6.update_layout(
        title='LSTM Model for Gross Withdrawals (MMcf)',
        xaxis_title='Date',
        yaxis_title='U.S. Natural Gas Gross Withdrawals (MMcf)',
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='right',
            x=1
        )
    )

    plot_html = fig6.to_html(full_html=False)
    pickle.dump(plot_html,open("forecast_lstm_prod_fig1.pkl","wb"))

    n_months = 60 #5 months prediction
    last_month = x_test[-1]
    predicted_prod = []
    for i in range(n_months):
        next_month = model7.predict(last_month.reshape(1, -1, 1))
        predicted_prod.append(next_month[0, 0])
        last_month = np.append(last_month[1:], next_month)

    predicted_prod = scaler.inverse_transform(np.array(predicted_prod).reshape(-1, 1))
    last_month_date = diff.index[-1] + pd.offsets.MonthEnd()
    predicted_dates = pd.date_range(start=last_month_date, periods=n_months+1, freq='M')[1:]
    predicted_df = pd.DataFrame({'U.S. Natural Gas Gross Withdrawals (MMcf)': predicted_prod.flatten()}, index=predicted_dates)

    df_pred = pd.concat([diff, predicted_df])

    fig7 = go.Figure()

    fig7.add_trace(go.Scatter(x=df_pred.index, y=df_pred['U.S. Natural Gas Gross Withdrawals (MMcf)'],
                        mode='lines', name='Actual'))

    fig7.add_trace(go.Scatter(x=predicted_df.index, y=predicted_df['U.S. Natural Gas Gross Withdrawals (MMcf)'],
                        mode='lines', name='Predicted'))

    fig7.update_layout(title='Forecasting of U.S. Natural Gas Gross Withdrawals (MMcf)', xaxis_title='Date', yaxis_title='Production')

    plot_html7 = fig7.to_html(full_html=False)
    pickle.dump(plot_html7,open("forecast_lstm_prod_fig2.pkl","wb"))

    #return render_template('forecast_lstm_prod.html', forecast_new = plot_html, forecast_new_ = plot_html7)
