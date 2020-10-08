#!/usr/bin/env bash
#  =============================================================================================
#                                    PURE PYTHON 3.7.0 VIRTUAL ENVIRONMENT
# ==============================================================================================

# virtualenvwrapper commands are not recognized WITHIN bash files. Not sure why.
# Need to source virtualenvwrapper.sh in order to be recognized.
# First: create a bash variable pointing to the path of virtualenvwrapper.sh
# name it: export VIRTUALENVWRAPPER_SH=/usr/local/bin/virtualenvwrapper.sh

# You can omit this line if you do not use virtual environment wrapper. You can create your virtual environment
# using pip and install the requirements.
source $VIRTUALENVWRAPPER_SH

# You can ignore this line as well. (same reason)
workon pytorch.1.4.0
pip freeze > requirements.txt

# Create a new V.E using virtualenvwrapper
# mkvirtualenv -r requirements.txt pytorch.1.2.0-dev
# mkvirtualenv -p $HOME/anaconda3/bin/python3 -r requirements.txt pytorch.1.2.0-dev

# Create a new V.E using virtualenv
# virtualenv ~/Venvs/pytorch.1.0.1

# Install requirements. Up to you for --no-index
# pip install --no-index -r requirements.txt
