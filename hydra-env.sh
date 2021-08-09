#!/bin/sh
set -x
mkdir -p "/tmp/$USER/miniconda3/envs/"
tar -C "/tmp/$USER/miniconda3/envs/" -xf hopfield-tracking.env.tar.lz4 -I lz4
