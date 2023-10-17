# switch to miniconda3 base as pytorch base uses python3.7
FROM continuumio/miniconda3:23.3.1-0

# permanent dependencies, put on top to avoid re-build
RUN pip install --upgrade pip
RUN apt-get update && apt-get install -y vim tmux && \
    pip install pip-tools
RUN pip install pandas==2.0.3 transformers==4.21.2 timm==0.9.2 einops==0.6.1 rasterio==1.3.7 tensorboard==2.13.0 datasets==2.14.5 tqdm==4.65.0 protobuf==4.24.3 colorama==0.4.6 scikit-learn==1.3.0 seaborn==0.12.2 matplotlib==3.7.2 && \
	pip uninstall torch torchaudio torchvision -y

WORKDIR /workspace

RUN chgrp -R 0 . && \
    chmod -R g=u .

RUN chgrp -R 0 /opt/conda && \
    chmod -R g=u /opt/conda

# tools
RUN apt-get update && apt-get install -y \
	vim \
	nmon

# put this at the end as we change this often, we add dummy steps to force rebuild the following lines when needed
# RUN pwd && pwd && pwd && pwd
RUN pip install --pre torch==2.0.1+cu117 torchvision --index-url https://download.pytorch.org/whl/cu117
RUN pip install torchmetrics==1.0.1
# RUN pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu117

RUN python3 -m pip install jupyterlab
EXPOSE 8889

ENTRYPOINT ["jupyter", "lab", "--port=8889", "--ip=0.0.0.0", "--allow-root"]
