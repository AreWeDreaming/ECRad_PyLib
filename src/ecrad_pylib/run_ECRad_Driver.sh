#!/bin/bash
if [ $# == 2 ]
  then
  python ECRad_Driver.py $1 $2
else
  python ECRad_Driver.py
fi

