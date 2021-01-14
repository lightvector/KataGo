from debian:buster

RUN apt-get update
RUN apt-get install -y g++ cmake libboost-filesystem-dev libeigen3-dev git ocl-icd-opencl-dev libboost-filesystem-dev libzip-dev


WORKDIR /KataGo
COPY . .

RUN rm CMakeCache.txt
RUN cmake . -DUSE_BACKEND=OPENCL -DNO_GIT_REVISION=1
RUN make
RUN strip katago

