FROM nvidia/cuda:11.0-cudnn8-devel-ubuntu18.04

ENV LD_LIBRARY_PATH /usr/local/cuda-11.0/lib64:/usr/local/cuda-11.0/extras/CUPTI/lib64:$LD_LIBRARY_PATH

# Ray freaks out if this isn't here
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

RUN apt update && apt install software-properties-common -y
# python3.8-dev includes headers that are needed to install pickle5 later
RUN apt-get install python3.8-dev -y
# install python 3.8 virtual environment
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt update && apt install python3.8-venv vim git -y
ENV VIRTUAL_ENV=/opt/venv
RUN python3.8 -m venv $VIRTUAL_ENV
ENV PATH=$VIRTUAL_ENV/bin:$PATH
RUN python -m pip install --upgrade pip setuptools

# install java, requirement to install hydra from source
RUN apt install default-jre -y

# Install AWS CLI
RUN apt install curl unzip -y && \
    curl "https://s3.amazonaws.com/aws-cli/awscli-bundle.zip" -o "awscli-bundle.zip" && \
    unzip awscli-bundle.zip && \
    ./awscli-bundle/install -i /usr/local/aws -b /usr/local/bin/aws

COPY datasets /datasets
# copy over source code
RUN mkdir -p /home/sam/Code/remote
WORKDIR /home/sam/Code/remote
COPY upcycle/ upcycle/
COPY gnosis/ gnosis/
COPY hydra/ hydra/
COPY olive-oil-ml/ olive-oil-ml/

# Install python dependencies
# scikit-learn dependencies (cython, numpy, scipy) have to be installed manually because pip is dumb.
# consider removing olive-oil-ml as a dependency
RUN python -m pip install --upgrade cython numpy scipy
RUN python -m pip install -e olive-oil-ml/
RUN python -m pip install -e upcycle/
RUN python -m pip install -r gnosis/requirements.txt
RUN python -m pip install -e gnosis/
RUN python -m pip install -e hydra/
RUN python -m pip install -e hydra/plugins/hydra_ray_launcher/
RUN python -m pip install -e hydra/plugins/hydra_ax_sweeper/

WORKDIR /home/sam/Code/remote/gnosis