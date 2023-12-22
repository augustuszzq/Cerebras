#!/bin/bash

# Clone the model zoo at the tested release
rm -r modelzoo
git clone https://github.com/Cerebras/modelzoo.git
# Pick commit that code has been previously run on
cd modelzoo
$ source venv_cerebras_pt/bin/activate
pip install -r requirements_cb_pt.txt
git checkout 4e2a4e02bd8969065cc54bd731d76887db0e45ad
cd ..


# Set up virtual environment
rm -r env
/software/cerebras/python3.7/bin/python3.7 -m venv env

# Add model zoo path to virtual environment
echo "$(pwd)/modelzoo" > env/lib/python3.7/site-packages/modelzoo.pth

# Install packages in virtual environment
source env/bin/activate
#pip3 install --disable-pip-version-check /opt/cerebras/wheels/old/cerebras_pytorch-1.8.0+de49801ca3-py3-none-any.whl --find-links=/opt/cerebras/wheels/
pip install --disable-pip-version-check /opt/cerebras/wheels/cerebras_pytorch-1.9.2+92b4fad15b-cp38-cp38-linux_x86_64.whl --find-links=/opt/cerebras/wheels
pip install numpy==1.23.4
pip install datasets transformers

