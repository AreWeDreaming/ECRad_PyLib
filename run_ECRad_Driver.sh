#!/bin/bash
source ../ECRad_core/set_environment.sh
if [$# == 2]
  then
  python ECRad_GUI_Driver.py $1 $2
else
  python ECRad_GUI_Driver.py
fi

