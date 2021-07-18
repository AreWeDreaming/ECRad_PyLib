#!/bin/tcsh

if ($HOSTNAME =~ *"mpg"* ) then
  module purge
  module load intel
  module load mkl
  module load texlive
  module load anaconda/3/2020.02
  module load git
  if ($?LD_LIBRARY_PATH) then
    setenv LD_LIBRARY_PATH $LD_LIBRARY_PATH\:$MKLROOT/lib/intel64/
  else
  	setenv LD_LIBRARY_PATH $MKLROOT/lib/intel64/
  endif
endif
rm id
git rev-parse HEAD > id
if ($# == 2) then
  python ECRad_GUI_Driver.py $1 $2
else
  python ECRad_GUI_Driver.py
endif
