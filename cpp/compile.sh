#!/bin/bash -eux

echo "This script is deprecated, use cmake instead"

/usr/bin/h5c++ -std=c++14 -fmessage-length=0 -Wall -Wno-sign-compare -Wno-strict-aliasing -Itclap-1.2.1/include -O3 write.cpp game/*.cpp dataio/*.cpp core/*.cpp -o write.exe

echo "Done"
