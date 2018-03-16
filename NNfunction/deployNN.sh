#!/bin/bash
rm digitNN.zip
zip -r digitNN.zip __main__.py mnist_loader.py network.py
wsk -i action delete digitnn
wsk -i action create digitnn --docker pulasthisupun/python2_nn_runtime digitNN.zip --timeout 300000
