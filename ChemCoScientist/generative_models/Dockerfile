FROM nvidia/cuda:11.4.3-runtime-ubuntu20.04
ENV TZ=Europe/Moscow
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-get update && apt-get install --no-install-recommends -y \
  build-essential \
  curl \
  wget \
  git \
  cmake \
  vim \
  pkg-config \
  unzip \
  libgtk2.0-dev \
  imagemagick \
  graphviz && \
  mkdir /projects

#RUN mkdir /projects/generative_models

COPY . /projects/generative_models

RUN mkdir -p ~/miniconda3 && \
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh && \
    bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3 && \
    rm -rf ~/miniconda3/miniconda.sh && \
    export PATH=~/miniconda3/miniconda/bin:$PATH && \
    ~/miniconda3/bin/conda init bash && \
    ~/miniconda3/bin/conda init zsh


ENV PATH ~/miniconda3/miniconda/bin:$PATH
ENV LB_LIBRARY_PATH="~/miniconda3/miniconda/lib:\$LB_LIBRARY_PATH"
SHELL ["/bin/bash", "-ic"]
RUN conda update -n base conda && \
    conda update --all -y
ENV PYTHONPATH="${PYTHONPATH}:/projects/generative_models"

RUN cd /projects/generative_models && \
    conda create -n Mol_gen_env python=3.8 && \
    conda deactivate && \
    conda activate Mol_gen_env && \ 
    conda install conda-libmamba-solver && \
    conda install -c "conda-forge/label/cf202003" openbabel --solver libmamba && \
    pip install -r requirements.txt && \
    pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu116 && \
    chmod 777 /projects/generative_models/api.sh


WORKDIR /projects/generative_models
ENTRYPOINT [ "/bin/bash", "/projects/generative_models/api.sh" ]

