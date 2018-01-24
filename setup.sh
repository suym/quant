#!/usr/bin/env bash

if [[ $# -eq 0 ]]; then
	printf "NAME\n\tsetup.sh - Main driver to set up the environment\n"
	printf "\nSYNOPSIS\n"
	printf "\n\t%-5s\n" "./setup.sh [OPTION]"
	printf "\nOPTIONS\n"
	printf "\n\t%-5s  %-40s\n"  "1"  "Softwares that have to be installed on your computer before using this package"
	printf "\n\t%-5s  %-40s\n"  "2"  "How to install these softwares"
fi

option=$1

case $option in
    1) echo "The following software needs to be installed....."
       echo "         "
       echo "Anaconda2"
       echo "tushare, minepy, pandas-datareader"
       echo "py-xgboost, TPOT, jupyter_contrib_nbextensions"
       echo "root_numpy, root_pandas"
       echo "         "
       ;;
    2) echo "How to install these softwares....."
       echo "         "
       echo "conda install py-xgboost"
       echo "pip install tushare, minepy, pandas-datareader, TPOT"
       echo "pip install root_numpy, root_pandas"
       echo "conda install -c conda-forge jupyter_contrib_nbextensions"
       echo "         "
       ;;
esac




