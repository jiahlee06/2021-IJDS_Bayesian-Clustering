# syntax=docker/dockerfile:1

FROM python:3.7-slim

WORKDIR /app

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

#COPY example.gms example.gms
COPY gamslice.txt /opt/gams/gams33.1_linux_x64_64_sfx/gamslice.txt

# Set GAMS version 
ENV GAMS_VERSION=33.1.0

# Set GAMS bit architecture
ENV GAMS_BIT_ARC=x64_64

# Install wget 
RUN apt-get update && apt-get install -y --no-install-recommends wget curl software-properties-common git unzip

# Download GAMS 
RUN curl -SL "https://d37drm4t2jghv5.cloudfront.net/distributions/${GAMS_VERSION}/linux/linux_${GAMS_BIT_ARC}_sfx.exe" --create-dirs -o /opt/gams/gams.exe

# Install GAMS 
RUN cd /opt/gams &&\
    chmod +x gams.exe; sync &&\
    ./gams.exe &&\
    rm -rf gams.exe 
    
ENV PATH "$PATH:/opt/gams/gams33.1_linux_x64_64_sfx"

# Run GAMS test 

#CMD [ "gams", "example.gms", "logOption=1", "--host=0.0.0.0" ]


