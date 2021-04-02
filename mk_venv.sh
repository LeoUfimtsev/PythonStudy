#!/bin/bash
python3  -m venv venv
# You may need to install requests for some bits.
# pip install requests
# check pip version with pip -V. Should be pip from python3 rather than 2.


# Due to bug in pycharm, pycharm can't use the python3 that is bundled with osx catalina:
# https://youtrack.jetbrains.com/issue/PY-38479
# Solution: Create env manually & point interpreter to env yourself.
