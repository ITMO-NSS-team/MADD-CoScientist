FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04
ENV TZ=Europe/Moscow
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-get update && apt-get install --no-install-recommends -y \
    python3.11 \
python3.11-venv \
python3.10 \
python3.10-venv \
pip \
git \
python3-dev \
build-essential \
curl \
pkg-config \
libgtk2.0-dev \
imagemagick \
graphviz \
cmake libboost-dev libomp-dev


# Create project directories
RUN mkdir -p /projects/BIAM_Chem

COPY ./requirements.txt /projects/BIAM_Chem/requirements.txt
COPY ./constraints.txt /projects/BIAM_Chem/constraints.txt

RUN python3.10 -m venv /projects/BIAM_Chem/venv310 && \
python3.11 -m venv /projects/BIAM_Chem/venv

WORKDIR /projects/BIAM_Chem

# Create and activate a virtual environment
ENV VIRTUAL_ENV=/projects/BIAM_Chem/venv

RUN /projects/BIAM_Chem/venv310/bin/pip install git+https://github.com/aimclub/FEDOT.git
RUN /projects/BIAM_Chem/venv310/bin/pip install git+https://github.com/aimclub/FEDOT.LLM.git@predict -c constraints.txt

ENV PATH="$VIRTUAL_ENV/bin:$PATH"
RUN /projects/BIAM_Chem/venv/bin/pip install -r requirements.txt 

COPY . /projects/BIAM_Chem/

RUN mv .env.example .env

# expose port
EXPOSE 8501

# start app
CMD ["/bin/bash", "-c", "source /projects/BIAM_Chem/venv/bin/activate && streamlit run streamlit_app.py --server.port=8501 --server.address=0.0.0.0"]