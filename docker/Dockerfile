#choose a fitfal version of pytorch image according to your cuda and modify the arguement TORCH_VERSION
#Reference website：https://hub.docker.com/r/pytorch/pytorch/tags
ARG TORCH_VERSION=1.9.0-cuda10.2-cudnn7-runtime
FROM pytorch/pytorch:${TORCH_VERSION}

# update & configure cuda and pip
RUN pip install --upgrade pip \
    & conda update -n base -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge conda \
    & conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/ \
    & conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/peterjc123/ \
    & conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/msys2/ \
    & conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge \
    & conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/ \
    & conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/ \
    & conda config --set show_channel_urls yes 

#download requirements
RUN pip install cvxopt \
    & conda install scipy \
    & pip install matplotlib \
    & pip install prettytable \
    & pip install ujson \
    & pip install pyyaml \
    & pip install pynvml \
    & pip install pandas 


#install flgo
RUN pip install flgo