#!/bin/bash
#suyumo

Dir=/besfs/users/suym/6.6.4.p01/Analysis/plot/python_doc/quant
Log=$Dir/run/log

if [[ $# -eq 0 ]]; then
    printf "NAME\n\tcheck.sh - Main driver to check programs\n"
    printf "\nSYNOPSIS\n"
    printf "\n\t%-5s\n" "./chk_pbsjobs.sh [OPTION]"
    printf "\nOPTIONS\n"
    printf "\n\t%-5s  %-40s\n"  "1"  "Check model_featureComp log files  "
    printf "\n\t%-5s  %-40s\n"  "2"  "Check model_IncrementalPCA log files  "
    printf "\n\t%-5s  %-40s\n"  "3"  "Check model_PCA log files  "
    printf "\n\t%-5s  %-40s\n"  "4"  "Check model_selection log files  "
    printf "\n\t%-5s  %-40s\n"  "4.1"  "Check model_cla_LinearSVC log files  "
    printf "\n\t%-5s  %-40s\n"  "4.2"  "Check model_cla_LogisticRegression log files  "
    printf "\n\t%-5s  %-40s\n"  "4.3"  "Check model_cla_RandomForestRegressor log files  "
    printf "\n\t%-5s  %-40s\n"  "4.4"  "Check model_cla_SVC log files  "
    printf "\n\t%-5s  %-40s\n"  "5"  "Check model_tpot log files  "
    printf "\n\t%-5s  %-40s\n"  "5.1"  "Check model_cla_tpot log files  "
    printf "\n\t%-5s  %-40s\n"  "5.2"  "Check model_reg_tpot log files  "
fi


option=$1

case $option in
    1) echo "Check model_featureComp log files..."
    cd $Log/model_featureComp
    cat model_featureComp.log
     ;;

    2) echo "Check model_IncrementalPCA log files..."
    cd $Log/model_IncrementalPCA
    cat model_IncrementalPCA.log
     ;;

    3) echo "Check model_PCA log files..."
    cd $Log/model_PCA
    cat model_PCA.log
     ;;

    4) echo "Check model_selection log files..."
	    ;;

    4.1) echo "Check model_cla_LinearSVC log files..."
    cd $Log/model_selection
    cat model_cla_LinearSVC.log
     ;;

    4.2) echo "Check model_cla_LogisticRegression log files..."
    cd $Log/model_selection
    cat model_cla_LogisticRegression.log
     ;;

    4.3) echo "Check model_cla_RandomForestRegressor log files..."
    cd $Log/model_selection
    cat model_cla_RandomForestRegressor.log
     ;;

    4.4) echo "Check model_cla_SVC log files..."
    cd $Log/model_selection
    cat model_cla_SVC.log
     ;;

    5) echo "Check model_tpot log files..."
	    ;;

    5.1) echo "Check model_cla_tpot log files..."
    cd $Log/model_tpot
    cat model_cla_tpot.log
     ;;

    5.2) echo "Check model_reg_tpot log files..."
    cd $Log/model_tpot
    cat model_reg_tpot.log
     ;;

esac

cd $Dir/chk_jobs
