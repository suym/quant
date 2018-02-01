#!/bin/bash
#suyumo

Dir_1=/besfs/users/suym/6.6.4.p01/Analysis/plot/python_doc/quant/run/inf_log
Dir_2=/besfs/users/suym/6.6.4.p01/Analysis/plot/python_doc/quant/run/log/model_selection
Dir_3=/besfs/users/suym/6.6.4.p01/Analysis/plot/python_doc/quant/run/log/model_IncrementalPCA
Dir_4=/besfs/users/suym/6.6.4.p01/Analysis/plot/python_doc/quant/chk_jobs

cd $Dir_1
rm cla_log.dat 
declare -a array=('GradientBoostingClassifier' 'LinearSVC' 'LogisticRegression' 'RandomForestClassifier' 'SVC_linear' 'SVC_rbf')

for num in ${array[@]}
do
   cd ${Dir_2}
   grep "Best parameters set found" model_cla_${num}.log |  sed "s/Best/${num}/">>${Dir_1}/cla_log.dat
   grep "The scores of test set" model_cla_${num}.log >>${Dir_1}/cla_log.dat

done

cd $Dir_3
grep "To DIY" model_IncrementalPCA.log |  sed 's/To/IncrementalPCA/'>>${Dir_1}/cla_log.dat

cd $Dir_4
