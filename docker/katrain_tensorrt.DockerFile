# set "Path to KataGo executable" as "./katago" in General and Engine settings
FROM nvcr.io/nvidia/tensorrt:23.04-py3
ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update
RUN apt-get install xclip -y
RUN apt-get install libsdl2-dev -y
RUN apt-get install zip -y
RUN apt-get install libmtdev-dev -y
RUN apt-get install ffmpeg -y
RUN apt-get install unzip -y
RUN apt-get install libzip5 -y
RUN pip install --upgrade pip
RUN pip install ffpyplayer
RUN pip install katrain
RUN wget https://github.com/lightvector/KataGo/releases/download/v1.14.1/katago-v1.14.1-trt8.6.1-cuda12.1-linux-x64.zip
RUN unzip ./katago-v1.14.1-trt8.6.1-cuda12.1-linux-x64.zip
#RUN rm ./katago-v1.14.1-trt8.6.1-cuda12.1-linux-x64.zip
CMD [ "katrain" ]