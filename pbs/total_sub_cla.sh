#!/bin/bash

Work_Dir=/besfs/users/suym/6.6.4.p01/Analysis/plot/python_doc/quant

cd $Work_Dir/pbs

./sub_cla_LinearSVC.sh
./sub_cla_LogisticRegression.sh
./sub_cla_RandomForestClassifier.sh
./sub_cla_SVC.sh
./sub_cla_GradientBoostingClassifier.sh

