#!/bin/bash

if [ -z "${PROJECT_ROOT_DIR}" ]; then
    SCRIPT_DIR=$( cd "$( dirname -- "${BASH_SOURCE[0]}" )/" &> /dev/null && pwd )
    source $SCRIPT_DIR/setup.sh
fi

logdir_rel="/home/rmenon/Desktop/dev/ml_results/aigo_results"
if [ $# -eq 1 ]; then
    logdir_rel="$1"
fi

logdir=$(cd $logdir_rel 2> /dev/null && pwd)
if [ $? -ne 0 ]; then
    echo "Invalid directory: $logdir_rel"
    exit 1
fi

echo Using logdir: "$logdir"
echo ""

tensorboard --logdir=$logdir
