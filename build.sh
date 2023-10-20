#!/bin/bash
python3 -m build -n -x
python3 -m pip install --no-deps .
git rev-parse HEAD > id