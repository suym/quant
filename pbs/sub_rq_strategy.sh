#!/bin/bash

Work_Dir=/besfs/users/suym/6.6.4.p01/Analysis/plot/python_doc/quant

Log_Dir=${Work_Dir}/run/log/rqalpha

Log_File=${Log_Dir}/rq_strategy.log
eLog_File=${Log_Dir}/rq_strategy.err


hep_sub -g physics -o ${Log_File}  -e ${eLog_File} -mem 5000 ${Work_Dir}/pbs/raw/model_rq_strategy.sh


