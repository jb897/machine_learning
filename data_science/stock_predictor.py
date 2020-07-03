## Stock price predictor based on Siraj Raval's youtube guide to data science

import numpy as np 
import pandas as pd
from sklearn.svm import SVR
import matplotlib.pyplot as plt 

# plt.switch_backend('newbackend')  

dates = []
prices = []

def predict_prices(dates, prices, x):
	dates = np.reshape(dates, (len(dates), 1))

	svr_lin = SVR(kernel='linear', C=1e3)
	svr_poly = SVR(kernel='poly', C=1e3, degree=2)
	svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
	## Gamma defines how far is the maximum distance (close to 0)
	svr_lin.fit(dates, prices)
	svr_poly.fit(dates, prices)
	svr_rbf.fit(dates, prices)

	plt.scatter(dates, prices, color='black', label='Data')
	plt.plot(dates, svr_rbf.predict(dates), color='red', label='RBF')
	plt.plot(dates, svr_lin.predict(dates), color='green', label='Linear Model')
	plt.plot(dates, svr_poly.predict(dates), color='blue', label='Polynomial Model')

	plt.xlabel('Date')
	plt.ylabel('Price')
	plt.title('Support Vector Regression')
	plt.legend()
	plt.show()

	return svr_rbf.predict(x)[0], svr_poly.predict(x)[0], svr_lin.predict(x)[0]



def get_data(filename):
	data = pd.read_csv(filename)
	
	## Get the dates, only the days of the month
	dates = data['Date'].values
	dates = [int(i.split('-')[2]) for i in dates]
	
	## Get the daily closing prices
	prices = data['Close'].values
	prices = [float(i) for i in prices]

	predicted_price = predict_prices(dates, prices, 29)

	return

get_data("TSLA.csv")