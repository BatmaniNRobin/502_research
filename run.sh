#!/bin/bash

make mpi
./main datasets/usethis.arff 5
./main datasets/segment-test.arff 5
./main datasets/segment-challenge.arff 5
./main datasets/small.arff 5
./main datasets/medium.arff 5
./main datasets/large.arff 5
