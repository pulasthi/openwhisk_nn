#!/bin/bash
wsk -i action delete testnn
wsk -i action create --docker pulasthisupun/python2_nn_runtime testnn testNN.py --timeout 300000 
