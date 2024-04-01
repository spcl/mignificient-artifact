FROM nvidia/cuda:11.6.1-cudnn8-devel-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y python3-dev cmake git libiberty-dev libattr1-dev wget libacl1-dev python3-pip
RUN wget https://github.com/Kitware/CMake/releases/download/v3.29.0/cmake-3.29.0-linux-x86_64.tar.gz && tar -xf cmake-3.29.0-linux-x86_64.tar.gz && mv cmake-3.29.0-linux-x86_64 /cmake && rm cmake-3.29.0-linux-x86_64.tar.gz
RUN pip install pybind11

