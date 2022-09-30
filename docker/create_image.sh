#!/bin/bash

cp ../requirements.txt .
sudo docker build -t hf_image .
rm requirements.txt