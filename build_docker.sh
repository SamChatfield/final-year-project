#!/bin/bash
docker pull samchatfield/tf-gpu:latest
docker build -t samchatfield/fyp-jupyter-gpu:latest .
