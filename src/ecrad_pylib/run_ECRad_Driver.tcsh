#!/bin/tcsh
source ../ECRad_core/set_environment.tcsh
if ( $# == 2 ) then
  python ECRad_Driver.py $1 $2
else
  python ECRad_Driver.py
endif
