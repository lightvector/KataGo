#!/bin/sh
wget https://github.com/ChinChangYang/KataGo/releases/download/v1.13.2-coreml1/kata1-b18c384nbt-s7709731328-d3715293823.bin.gz
mv kata1-b18c384nbt-s7709731328-d3715293823.bin.gz DerivedData/KataGo/Build/Products/Debug/model.bin.gz
wget https://github.com/ChinChangYang/KataGo/releases/download/v1.13.2-coreml1/KataGoModel19x19fp16v14s7709731328.mlpackage.zip
mv KataGoModel19x19fp16v14s7709731328.mlpackage.zip DerivedData/KataGo/Build/Products/Debug/
unzip DerivedData/KataGo/Build/Products/Debug/KataGoModel19x19fp16v14s7709731328.mlpackage.zip -d DerivedData/KataGo/Build/Products/Debug/
rm -rf DerivedData/KataGo/Build/Products/Debug/KataGoModel19x19fp16.mlpackage
mv DerivedData/KataGo/Build/Products/Debug/KataGoModel19x19fp16v14s7709731328.mlpackage DerivedData/KataGo/Build/Products/Debug/KataGoModel19x19fp16.mlpackage
ln -s ../../../../../../configs/misc/coreml_example.cfg DerivedData/KataGo/Build/Products/Debug/gtp.cfg
ln -s ../../../../../../tests DerivedData/KataGo/Build/Products/Debug/tests
ln -s ../Debug/model.bin.gz DerivedData/KataGo/Build/Products/Release/
ln -s ../Debug/KataGoModel19x19fp16.mlpackage DerivedData/KataGo/Build/Products/Release/
ln -s ../Debug/gtp.cfg DerivedData/KataGo/Build/Products/Release/
