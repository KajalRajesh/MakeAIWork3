#!/usr/bin/env bash

function installWithoutConda {
  echo "Install without conda"

  echo "Install requierments with pip"
  python -m pip install --upgrade pip --no-cache-dir -r install/pip/no_conda.txt

}  

# Install all required libraries t
function installWithPip {
  echo "Install with pip"

  echo "Prepare pip"
  python -m pip install --upgrade pip    
  python -m pip install setuptools
  python -m pip install -U sentence-transformers
  echo "Install requierments with pip"
  python -m pip install --no-cache-dir -r install/pip/requirements.txt

}

installWithoutConda
installWithPip