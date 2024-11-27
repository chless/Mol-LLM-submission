FROM nvidia/cuda:12.2.2-cudnn8-devel-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ARG PYTHON_VERSION=3.10
ENV PATH=/miniconda/bin:${PATH}

# Install dependencies
RUN apt-get update && apt-get install locales -y
RUN locale-gen en_US.UTF-8
RUN apt-get update \
    && apt-get install -y python3-pip python3-dev golang-1.18 git wget curl zsh tmux vim \
    && rm -rf /var/lib/apt/lists/*
RUN ln -s /usr/bin/python3 /usr/bin/python
RUN sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)"

ARG HOME=/root
WORKDIR $HOME
RUN git clone https://github.com/gpakosz/.tmux.git
RUN ln -s -f .tmux/.tmux.conf
RUN cp .tmux/.tmux.conf.local .
RUN echo "set-option -g default-shell /bin/zsh" >> .tmux.conf.local
RUN echo "set-option -g history-limit 10000" >> .tmux.conf.local


RUN useradd -ms /bin/zsh github-action

RUN apt-get update \
    && apt-get install -y clang-format clang-tidy swig qtdeclarative5-dev \
    && rm -rf /var/lib/apt/lists/*

# Install conda
RUN curl -LO https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    bash Miniconda3-latest-Linux-x86_64.sh -p /miniconda -b && \
    rm Miniconda3-latest-Linux-x86_64.sh && \
    conda update -y conda


RUN conda install --quiet --yes python=${PYTHON_VERSION} && \
    conda clean --yes --all


# RUN conda install -y pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia && \
    # conda install -y pyg=*=*cu* -c pyg && \
    # conda clean -y --all

# Upgrade pip, install py libs
RUN pip install --upgrade pip

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .

# PyTorch
RUN pip install torch torchvision torchaudio
RUN pip install --upgrade numpy thinc spacy opencv-python


RUN printf "\nexport PATH=/miniconda/bin:${PATH}" >> /root/.zshrc
RUN echo 'export SHELL=/bin/zsh' >> ~/.bash_profile
RUN echo 'exec /bin/zsh -l' >> ~/.bash_profile