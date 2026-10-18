FROM nvidia/cuda:12.2.2-cudnn8-devel-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ARG PYTHON_VERSION=3.10
ENV PATH=/miniconda/bin:${PATH}
ARG HOME=/root

RUN apt-get update && apt-get install -y \
    locales \
    python3-pip python3-dev \
    golang-1.18 \
    git wget curl \
    zsh tmux vim htop \
    clang-format clang-tidy \
    swig \
    qtdeclarative5-dev && \
    rm -rf /var/lib/apt/lists/*

RUN locale-gen en_US.UTF-8 && \
    ln -s /usr/bin/python3 /usr/bin/python

RUN sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)"

WORKDIR $HOME
RUN git clone https://github.com/gpakosz/.tmux.git && \
    ln -s -f .tmux/.tmux.conf && \
    cp .tmux/.tmux.conf.local . && \
    echo "set-option -g default-shell /bin/zsh" >> .tmux.conf.local && \
    echo "set-option -g history-limit 10000" >> .tmux.conf.local

RUN git clone https://github.com/zsh-users/zsh-syntax-highlighting.git ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-syntax-highlighting && \
    git clone https://github.com/zsh-users/zsh-autosuggestions ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-autosuggestions && \
    sed -i '/^plugins=/c\plugins=(git zsh-autosuggestions zsh-syntax-highlighting zsh-completions)' ~/.zshrc && \
    sed -i 's/^ZSH_THEME=".*"/ZSH_THEME="juanghurtado"/' ~/.zshrc

RUN useradd -ms /bin/zsh github-action

RUN curl -LO https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    bash Miniconda3-latest-Linux-x86_64.sh -p /miniconda -b && \
    rm Miniconda3-latest-Linux-x86_64.sh && \
    conda update -y conda

RUN conda install --quiet --yes python=${PYTHON_VERSION} && \
    conda clean --yes --all

RUN pip install --upgrade pip

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .

RUN pip install torch torchvision torchaudio flash_attn && \
    pip install --upgrade numpy thinc spacy opencv-python

RUN printf "\nexport PATH=/miniconda/bin:${PATH}" >> /root/.zshrc && \
    echo 'export SHELL=/bin/zsh' >> ~/.bash_profile && \
    echo 'exec /bin/zsh -l' >> ~/.bash_profile