FROM nvidia/cuda:9.2-cudnn7-devel

RUN apt-get update && apt-get install -y --no-install-recommends \
    sudo \
    git \
    zip \
    libopencv-dev \
    build-essential libssl-dev libbz2-dev libreadline-dev libsqlite3-dev curl \
    wget ca-certificates cmake && \
    rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

ARG UID
RUN useradd docker -u $UID -G sudo -s /bin/bash -m
RUN echo 'Defaults visiblepw' >> /etc/sudoers
RUN echo 'docker ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers

USER docker

ENV PYENV_ROOT /home/docker/.pyenv
ENV PATH $PYENV_ROOT/shims:$PYENV_ROOT/bin:$PATH
RUN curl -L https://raw.githubusercontent.com/pyenv/pyenv-installer/master/bin/pyenv-installer | bash

ENV PYTHON_VERSION 3.6.0
RUN pyenv install ${PYTHON_VERSION} && pyenv global ${PYTHON_VERSION}

RUN pip install -U pip setuptools

COPY requirements.txt /tmp/requirements.txt
RUN CHAINER_BUILD_CHAINERX=1 CHAINERX_BUILD_CUDA=1 \
    pip install --no-cache-dir -r /tmp/requirements.txt

# Setup jupyter
RUN jupyter notebook --generate-config && \
    ipython profile create

RUN echo "c.NotebookApp.ip = '0.0.0.0'" >> /home/docker/.jupyter/jupyter_notebook_config.py && \
    echo "c.NotebookApp.open_browser = False" >> /home/docker/.jupyter/jupyter_notebook_config.py && \
    echo "c.InteractiveShellApp.matplotlib = 'inline'" >> /home/docker/.ipython/profile_default/ipython_config.py && \
    HASH=$(python3 -c "from IPython.lib import passwd; print(passwd('passw0rd'))") && \
    echo "c.NotebookApp.password = u'${HASH}'" >> /home/docker/.jupyter/jupyter_notebook_config.py && \
    echo "c.NotebookApp.allow_root = True" >> /home/docker/.jupyter/jupyter_notebook_config.py

# Specify matplotlib backend
RUN mkdir -p /home/docker/.config/matplotlib
RUN echo "backend : Agg" >> /home/docker/.config/matplotlib/matplotlibrc

WORKDIR /work

ENTRYPOINT ["/bin/bash"]
