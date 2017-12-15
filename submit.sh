#!/usr/bin/env bash

# Main driver to submit jobs 


usage() {
    printf "NAME\n\tsubmit.sh - Main driver to submit jobs\n"
    printf "\nSYNOPSIS\n"
    printf "\n\t%-5s\n" "./submit.sh [OPTION]" 
    printf "\nOPTIONS\n" 
    printf "\n\t%-5s  %-40s\n"  "0.1"      "[Stock]" 
    printf "\n\t%-5s  %-40s\n"  "0.1.1"    "Get stock data from tushare" 
    printf "\n\n" 
}


if [[ $# -eq 0 ]]; then
    usage
fi


option=$1

case $option in 
    0.1) echo "Stock ..."
	 ;;

    0.1.1) echo "Get stock data from tushare ..."
		cd ./analysis/get_data/
		python get_hs300_data.py
		cd ../../
	   ;;

esac

