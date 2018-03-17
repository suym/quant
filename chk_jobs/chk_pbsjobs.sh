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
    printf "\n\t%-5s  %-40s\n"  "4.1.1"  "Check model_cla_LinearSVC log files  "
    printf "\n\t%-5s  %-40s\n"  "4.1.2"  "Check model_cla_LogisticRegression log files  "
    printf "\n\t%-5s  %-40s\n"  "4.1.3"  "Check model_cla_RandomForestClassifier log files  "
    printf "\n\t%-5s  %-40s\n"  "4.1.4"  "Check model_cla_SVC_linear log files  "
    printf "\n\t%-5s  %-40s\n"  "4.1.5"  "Check model_cla_SVC_rbf log files  "
    printf "\n\t%-5s  %-40s\n"  "4.1.6"  "Check model_cla_GradientBoostingClassifier log files  "
    printf "\n\t%-5s  %-40s\n"  "4.2.1"  "Check model_reg_Lasso log files  "
    printf "\n\t%-5s  %-40s\n"  "4.2.2"  "Check model_reg_Ridge log files  "
    printf "\n\t%-5s  %-40s\n"  "4.2.3"  "Check model_reg_GradientBoostingRegressor_huber log files  "
    printf "\n\t%-5s  %-40s\n"  "4.2.4"  "Check model_reg_GradientBoostingRegressor_lslad log files  "
    printf "\n\t%-5s  %-40s\n"  "4.2.5"  "Check model_reg_LinearSVR log files  "
    printf "\n\t%-5s  %-40s\n"  "4.2.6"  "Check model_reg_RandomForestRegressor log files  "
    printf "\n\t%-5s  %-40s\n"  "4.2.7"  "Check model_reg_SVR_linear log files  "
    printf "\n\t%-5s  %-40s\n"  "4.2.8"  "Check model_reg_SVR_rbf log files  "
    printf "\n\t%-5s  %-40s\n"  "4.2.9"  "Check model_reg_LSTM log files  "
    printf "\n\t%-5s  %-40s\n"  "4.2.10"  "Check model_reg_MFNN log files  "
    printf "\n\t%-5s  %-40s\n"  "5"  "Check model_tpot log files  "
    printf "\n\t%-5s  %-40s\n"  "5.1"  "Check model_cla_tpot log files  "
    printf "\n\t%-5s  %-40s\n"  "5.2"  "Check model_reg_tpot log files  "
    printf "\n\t%-5s  %-40s\n"  "6"  "Check store_data log files  "
    printf "\n\t%-5s  %-40s\n"  "6.1"  "Check store_data_reg_model log files  "
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

    4.1.1) echo "Check model_cla_LinearSVC log files..."
    cd $Log/model_selection
    cat model_cla_LinearSVC.log
     ;;

    4.1.2) echo "Check model_cla_LogisticRegression log files..."
    cd $Log/model_selection
    cat model_cla_LogisticRegression.log
     ;;

    4.1.3) echo "Check model_cla_RandomForestClassifier log files..."
    cd $Log/model_selection
    cat model_cla_RandomForestClassifier.log
     ;;

    4.1.4) echo "Check model_cla_SVC_linear log files..."
    cd $Log/model_selection
    cat model_cla_SVC_linear.log
     ;;

    4.1.5) echo "Check model_cla_SVC_rbf log files..."
    cd $Log/model_selection
    cat model_cla_SVC_rbf.log
     ;;

    4.1.6) echo "Check model_cla_GradientBoostingClassifier log files..."
    cd $Log/model_selection
    cat model_cla_GradientBoostingClassifier.log
     ;;

    4.2.1) echo "Check model_reg_Lasso log files..."
    cd $Log/model_selection
    cat model_reg_Lasso.log
     ;;

    4.2.2) echo "Check model_reg_Ridge log files..."
    cd $Log/model_selection
    cat model_reg_Ridge.log
     ;;

    4.2.3) echo "Check model_reg_GradientBoostingRegressor_huber log files..."
    cd $Log/model_selection
    cat model_reg_GradientBoostingRegressor_huber.log
     ;;

    4.2.4) echo "Check model_reg_GradientBoostingRegressor_lslad log files..."
    cd $Log/model_selection
    cat model_reg_GradientBoostingRegressor_lslad.log
     ;;

    4.2.5) echo "Check model_reg_LinearSVR log files..."
    cd $Log/model_selection
    cat model_reg_LinearSVR.log
     ;;

    4.2.6) echo "Check model_reg_RandomForestRegressor log files..."
    cd $Log/model_selection
    cat model_reg_RandomForestRegressor.log
     ;;

    4.2.7) echo "Check model_reg_SVR_linear log files..."
    cd $Log/model_selection
    cat model_reg_SVR_linear.log
     ;;

    4.2.8) echo "Check model_reg_SVR_rbf log files..."
    cd $Log/model_selection
    cat model_reg_SVR_rbf.log
     ;;

    4.2.9) echo "Check model_reg_LSTM log files..."
    cd $Log/model_selection
    cat model_reg_LSTM.log
     ;;

    4.2.10) echo "Check model_reg_MFNN log files..."
    cd $Log/model_selection
    cat model_reg_MFNN.log
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

    6) echo "Check store_data log files..."
	    ;;

    6.1) echo "Check store_data_reg_model log files..."
    cd $Log/store_data
    cat store_data_reg_model.log
     ;;


esac

cd $Dir/chk_jobs
