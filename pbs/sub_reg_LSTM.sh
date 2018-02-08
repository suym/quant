#!/bin/bash

Work_Dir=/besfs/users/suym/6.6.4.p01/Analysis/plot/python_doc/quant

Log_Dir=${Work_Dir}/run/log/model_selection

Log_File=${Log_Dir}/model_reg_LSTM.log
eLog_File=${Log_Dir}/model_reg_LSTM.err


hep_sub -g physics -o ${Log_File}  -e ${eLog_File} -mem 5000 ${Work_Dir}/pbs/raw/model_reg_LSTM.sh



