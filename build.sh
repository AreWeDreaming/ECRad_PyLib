#!/bin/bash
$PYTHON -m build -n -x
$PYTHON -m pip install --no-deps .
git rev-parse HEAD > id