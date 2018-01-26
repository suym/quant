#!/bin/bash

Work_Dir=/besfs/users/suym/6.6.4.p01/Analysis/plot/python_doc/quant

cd $Work_Dir/pbs

./sub_cla_LinearSVC.sh
./sub_cla_LogisticRegression.sh
./sub_cla_RandomForestRegressor.sh
./sub_cla_SVC.sh
./sub_cla_tpot.sh
./sub_featureComp.sh
./sub_IncrementalPCA_job.sh
./sub_PCA_job.sh


