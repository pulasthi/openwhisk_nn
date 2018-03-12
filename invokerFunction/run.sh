#!/bin/bash
wsk -i action delete $1
wsk -i action create --docker pulasthisupun/python2_nn_runtime $1 $2
wsk -i action invoke $1 --param layers '[784, 8, 10]'
echo wsk -i activation get $id
