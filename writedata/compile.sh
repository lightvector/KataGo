#!/bin/bash -eux

#/usr/bin/h5c++ -std=c++14 -fmessage-length=0 -O3 example.cpp -o example.exe

/usr/bin/h5c++ -std=c++14 -fmessage-length=0 -O3 write.cpp fastboard.cpp core/*.cpp -o write.exe
