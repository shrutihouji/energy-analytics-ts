from flask import Flask, render_template
from data_visual import display_data_price, display_data_prod, display_data_import_exports, display_data_consumptions, input_price, input_prod, plot1,plot2,prod_plot1, prod_plot2, uni_plot2, uni_plot3, con_plot1, con_plot2, auto_plot1, auto_plot2, prod_plot3, prod_plot4
from data_visual import forecast_lstm_price ,  forecast_lstm_prod, forecast_var_con, forecast_var_impexp

import pickle
import pandas as pd
import numpy as np
import math
from math import sqrt
from numpy import array
import yfinance as yf

import plotly.express as px
import plotly.graph_objs as go
import plotly.io as pio
from sklearn.preprocessing import MinMaxScaler 
import matplotlib.pyplot as plt

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import pacf
from statsmodels.tsa.stattools import acf

import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.optimizers import SGD
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense 

app=Flask(__name__)
@app.route("/")
def home():
    return render_template("main_new.html")

@app.route('/Dashboard/Resources')
def hello():
    data1 = display_data_price()
    data2 = display_data_prod()
    data3 = display_data_import_exports()
    data4 = display_data_consumptions()
    return render_template('index.html', returnList1 = data1,returnList2 = data2, returnList3 = data3, returnList4 = data4 )

@app.route('/Dashboard/Resources/Visualizations')
def visual():
    #visual1=plot1()
    #plot1()
    with open('plot1.pkl','rb') as f:
        visual1 = pickle.load(f)
    #uni_plot3()
    with open('uni_plot3.pkl','rb') as f:
        visual2 = pickle.load(f)
    #auto_plot1()
    with open('auto_plot1.pkl','rb') as f:
        visual3 = pickle.load(f)
    #auto_plot2()
    with open('auto_plot2.pkl','rb') as f:
        visual4 = pickle.load(f)
    return render_template('visuals.html', visual_1=visual1, visual_2=visual2,visual_3=visual3,visual_4=visual4)

@app.route('/Dashboard/Resources/Visualizations1')
def visual2():
    #visual1=uni_plot2()
    #uni_plot2()
    with open('uni_plot2.pkl','rb') as f:
        visual1 = pickle.load(f)
    #visual2=plot2()
    #plot2()
    with open('plot2.pkl','rb') as f:
        visual2 = pickle.load(f)

    return render_template('visuals1.html', visual_1=visual1, visual_2=visual2)
 
#VAR MODEL

@app.route('/Dashboard/Resources/Forecasting1')
def forecast_plot1():
    #forecast_var_impexp()

    with open('var_impexp_fig1.pkl','rb') as f:
        plot_html = pickle.load(f)
    
    with open('var_impexp_fig2.pkl','rb') as f:
        plot_html1 = pickle.load(f)
 
    return render_template('forecast_new.html', forecast_new=plot_html, forecast_new2=plot_html1)

#NATURAL GAS - PRODUCTION
@app.route('/Dashboard/Resources/Visualizations_Production')
def visual_production():
    #visual1=prod_plot1()
    #prod_plot1()
    with open('prod_plot1.pkl','rb') as f:
        visual1 = pickle.load(f)
    #visual2=prod_plot2()
    #prod_plot2()
    with open('prod_plot2.pkl','rb') as f:
        visual2 = pickle.load(f)
    #visual3=prod_plot3()
    #prod_plot3()
    with open('prod_plot3.pkl','rb') as f:
        visual3 = pickle.load(f)
    #visual4=prod_plot4()
    #prod_plot4()
    with open('prod_plot4.pkl','rb') as f:
        visual4 = pickle.load(f)
    return render_template('visuals_production.html', visual_1=visual1, visual_2=visual2, visual_3=visual3,visual_4=visual4)

#natural Gas-Consumption
@app.route('/Dashboard/Resources/Visualizations_Consumption')
def visual_consumption():
    # visual1=con_plot1()
    #con_plot1()
    with open('con_plot1.pkl','rb') as f:
        visual1 = pickle.load(f)
    # visual2=con_plot2()
    #con_plot2()
    with open('con_plot2.pkl','rb') as f:
        visual2 = pickle.load(f)
    return render_template('visuals_consumption.html', visual_1=visual1, visual_2=visual2)

#VAR MODEL-Consumption

@app.route('/Dashboard/Resources/Forecasting_Consumption')
def forecast_plot_con():
    #forecast_var_con()

    with open('var_con_fig1.pkl','rb') as f:
        plot_html = pickle.load(f)
    
    with open('var_con_fig2.pkl','rb') as f:
        plot_html1 = pickle.load(f)
 
    return render_template('forecast_consumption.html', forecast1=plot_html, forecast2=plot_html1)

#LSTM Model 
@app.route('/Dashboard/Resources/Forecasting_LSTM_price')
def forecast_LSTM_price():
    #forecast_lstm_price()

    with open('forecast_lstm_price_fig1.pkl','rb') as f:
        plot_html = pickle.load(f)
    
    with open('forecast_lstm_price_fig2.pkl','rb') as f:
        plot_html7 = pickle.load(f)

    return render_template('forecast_lstm_price.html', forecast_new = plot_html, forecast_new_ = plot_html7)

@app.route('/Dashboard/Resources/Forecasting_LSTM_prod')
def forecast_LSTM_prod():
    #forecast_lstm_prod()

    with open('forecast_lstm_prod_fig1.pkl','rb') as f:
        plot_html = pickle.load(f)
    
    with open('forecast_lstm_prod_fig2.pkl','rb') as f:
        plot_html7 = pickle.load(f)

    return render_template('forecast_lstm_prod.html', forecast_new = plot_html, forecast_new_ = plot_html7)
