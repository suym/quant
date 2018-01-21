#!/usr/bin/python
# -*- coding: utf-8 -*-

# model_featureComp.py

from sklearn.linear_model import (LinearRegression, Ridge, 
                                  Lasso, RandomizedLasso)
from sklearn.feature_selection import RFE, f_regression
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import pandas as pd
import datetime as dt
from minepy import MINE
from tool import create_lagged_series


def rank_to_list(ranks, order=1):
    minmax = MinMaxScaler()
    ranks = minmax.fit_transform(order*np.array([ranks]).T).T[0]
    ranks = map(lambda x: round(x, 2), ranks)
    return ranks


if __name__ == "__main__":
	# Create a lagged series of the hs300 stock market index
	snpret = create_lagged_series(
        "hs300",
        dt.datetime(2016, 1, 30),
        dt.datetime(2017, 12, 31)
	)
	# Standardized features
	x_ori = snpret.drop(['price_change', 'cla_Direction','reg_Direction'], axis = 1)
	scaler = StandardScaler().fit(x_ori)
	X = scaler.transform(x_ori)
	X = pd.DataFrame(X, index = x_ori.index, columns = x_ori.columns)
	Y = snpret["cla_Direction"]
	
	ranks=pd.DataFrame(index=X.columns)

	lr = LinearRegression(normalize=True)
	lr.fit(X, Y)
	ranks["Linear reg"] = rank_to_list(np.abs(lr.coef_))

	ridge = Ridge(alpha=7)
	ridge.fit(X, Y)
	ranks["Ridge"] = rank_to_list(np.abs(ridge.coef_))

	lasso = Lasso(alpha=.05)
	lasso.fit(X, Y)
	ranks["Lasso"] = rank_to_list(np.abs(lasso.coef_))

	rlasso = RandomizedLasso(alpha=0.04)
	rlasso.fit(X, Y)
	ranks["Stability"] = rank_to_list(np.abs(rlasso.scores_))
	
	#stop the search when 5 features are left (they will get equal scores)
	rfe = RFE(lr, n_features_to_select=5)
	rfe.fit(X,Y)
	ranks["RFE"] = rank_to_list(map(float, rfe.ranking_), order=-1)

	rf = RandomForestRegressor()
	rf.fit(X,Y)
	ranks["RF"] = rank_to_list(rf.feature_importances_)

	f, pval  = f_regression(X, Y, center=True)
	ranks["Corr."] = rank_to_list(f)

	mine = MINE()
	mic_scores = []
	for i in range(X.shape[1]):
		mine.compute_score(X.iloc[:,i], Y)
		m = mine.mic()
		mic_scores.append(m)

	ranks["MIC"] = rank_to_list(mic_scores)

	print ranks



