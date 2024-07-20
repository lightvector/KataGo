#!/bin/sh
wget https://github.com/ChinChangYang/KataGo/releases/download/v1.13.2-coreml1/kata1-b18c384nbt-s7709731328-d3715293823.bin.gz
mv kata1-b18c384nbt-s7709731328-d3715293823.bin.gz DerivedData/KataGo/Build/Products/Debug/model.bin.gz
wget https://github.com/lightvector/KataGo/releases/download/v1.4.5/g170-b40c256x2-s5095420928-d1229425124.bin.gz
mv g170-b40c256x2-s5095420928-d1229425124.bin.gz DerivedData/KataGo/Build/Products/Debug/modelv8.bin.gz
wget https://github.com/ChinChangYang/KataGo/releases/download/v1.13.2-coreml1/KataGoModel19x19fp16v14s7709731328.mlpackage.zip
mv KataGoModel19x19fp16v14s7709731328.mlpackage.zip DerivedData/KataGo/Build/Products/Debug/
rm -rf DerivedData/KataGo/Build/Products/Debug/KataGoModel19x19fp16.mlpackage
unzip DerivedData/KataGo/Build/Products/Debug/KataGoModel19x19fp16v14s7709731328.mlpackage.zip -d DerivedData/KataGo/Build/Products/Debug/
mv DerivedData/KataGo/Build/Products/Debug/KataGoModel19x19fp16v14s7709731328.mlpackage DerivedData/KataGo/Build/Products/Debug/KataGoModel19x19fp16.mlpackage
wget https://github.com/ChinChangYang/KataGo/releases/download/v1.15.1-coreml1/KataGoModel19x19fp32meta1.mlpackage.zip
mv KataGoModel19x19fp32meta1.mlpackage.zip DerivedData/KataGo/Build/Products/Debug/
rm -rf DerivedData/KataGo/Build/Products/Debug/KataGoModel19x19fp32meta1.mlpackage
unzip DerivedData/KataGo/Build/Products/Debug/KataGoModel19x19fp32meta1.mlpackage.zip -d DerivedData/KataGo/Build/Products/Debug/
ln -s ../../../../../../configs/misc/coreml_example.cfg DerivedData/KataGo/Build/Products/Debug/gtp.cfg
ln -s ../../../../../../configs/misc/metal_gtp.cfg DerivedData/KataGo/Build/Products/Debug/metal_gtp.cfg
ln -s ../../../../../../tests DerivedData/KataGo/Build/Products/Debug/tests
ln -s ../Debug/model.bin.gz DerivedData/KataGo/Build/Products/Release/
ln -s ../Debug/KataGoModel19x19fp16.mlpackage DerivedData/KataGo/Build/Products/Release/
ln -s ../Debug/gtp.cfg DerivedData/KataGo/Build/Products/Release/
