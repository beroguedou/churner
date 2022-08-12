#!/bin/bash

if [[ $CHURNERSETTER == "true" ]]
then
    echo "The PYTHONPATH variable is already set !"
else
    OUTPUT="$(echo `pwd`)"
    export PYTHONPATH="${PYTHONPATH}:${OUTPUT}"
    echo "The PYTHONPATH variable is set to $PYTHONPATH"
    export CHURNERSETTER="true"
fi


