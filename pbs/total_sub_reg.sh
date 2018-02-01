#!/bin/bash

Work_Dir=/besfs/users/suym/6.6.4.p01/Analysis/plot/python_doc/quant

cd $Work_Dir/pbs

./sub_reg_GradientBoostingRegressor_huber.sh
./sub_reg_GradientBoostingRegressor_lslad.sh
./sub_reg_Lasso.sh
./sub_reg_LinearSVR.sh
./sub_reg_RandomForestRegressor.sh
./sub_reg_Ridge.sh
./sub_reg_SVR_linear.sh
./sub_reg_SVR_rbf.sh

