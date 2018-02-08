#!/bin/bash
#suyumo

Dir_1=/besfs/users/suym/6.6.4.p01/Analysis/plot/python_doc/quant/run/inf_log
Dir_2=/besfs/users/suym/6.6.4.p01/Analysis/plot/python_doc/quant/run/backup_log
Dir_3=/besfs/users/suym/6.6.4.p01/Analysis/plot/python_doc/quant/chk_jobs

if [[ $# -eq 0 ]]; then
    printf "NAME\n\tbackup_log.sh - Main driver to backup_log programs\n"
    printf "\nSYNOPSIS\n"
    printf "\n\t%-5s\n" "./backup_log.sh [OPTION]"
    printf "\nOPTIONS\n"
    printf "\n\t%-5s  %-40s\n"  "1"  "back up cla log files "
    printf "\n\t%-5s  %-40s\n"  "1.1"  "back up cla log files for the first iteration "
    printf "\n\t%-5s  %-40s\n"  "1.2"  "back up cla log files for the second iteration "
    printf "\n\t%-5s  %-40s\n"  "2"  "back up reg log files "
    printf "\n\t%-5s  %-40s\n"  "2.1"  "back up reg log files for the first iteration "
    printf "\n\t%-5s  %-40s\n"  "2.2"  "back up reg log files for the second iteration "
fi

option=$1

case $option in
    1) echo "back up cla log files..."
            ;;

    1.1) echo "back up cla log files for the first iteration..."
    cd $Dir_2/
    rm cla_log_first.dat -f
    cp $Dir_1/cla_log.dat .
    mv cla_log.dat cla_log_first.dat
    cat cla_log_first.dat
     ;;

    1.2) echo "back up cla log files for the second iteration..."
    cd $Dir_2/
    rm cla_log_socond.dat -f
    cp $Dir_1/cla_log.dat .
    mv cla_log.dat cla_log_second.dat
    cat cla_log_second.dat
     ;;

    2) echo "back up reg log files..."
            ;;

    2.1) echo "back up reg log files for the first iteration..."
    cd $Dir_2/
    rm reg_log_first.dat -f
    cp $Dir_1/reg_log.dat .
    mv reg_log.dat reg_log_first.dat
    cat reg_log_first.dat
     ;;

    2.2) echo "back up reg log files for the second iteration..."
    cd $Dir_2/
    rm reg_log_second.dat -f
    cp $Dir_1/reg_log.dat .
    mv reg_log.dat reg_log_second.dat
    cat reg_log_second.dat
     ;;

esac



