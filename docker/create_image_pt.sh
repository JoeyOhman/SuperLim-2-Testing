#!/bin/bash

cp ../requirements.txt .
sudo docker build -t joeyohman/pt_ray_joey:latest -f Dockerfile_pt .
rm requirements.txt