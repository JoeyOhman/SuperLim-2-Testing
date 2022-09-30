FROM huggingface/transformers-pytorch-gpu
# COPY ../requirements.txt .
WORKDIR /workdir/
COPY requirements.txt /workdir/requirements.txt
RUN pip install -r requirements.txt
# WORKDIR /workdir
EXPOSE 13377