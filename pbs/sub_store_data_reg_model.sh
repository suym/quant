#!/bin/bash

Work_Dir=/besfs/users/suym/6.6.4.p01/Analysis/plot/python_doc/quant

Log_Dir=${Work_Dir}/run/log/store_data

Log_File=${Log_Dir}/store_data_reg_model.log
eLog_File=${Log_Dir}/store_data_reg_model.err


hep_sub -g physics -o ${Log_File}  -e ${eLog_File} -mem 5000 ${Work_Dir}/pbs/raw/store_data_reg_model.sh



