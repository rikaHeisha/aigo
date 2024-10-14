#!/bin/bash

if [ -z "${PROJECT_ROOT_DIR}" ]; then
    SCRIPT_DIR=$( cd "$( dirname -- "${BASH_SOURCE[0]}" )/" &> /dev/null && pwd )
    source $SCRIPT_DIR/setup.sh
fi

jupyter-notebook --no-browser
