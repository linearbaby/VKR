FROM python:3.10.6

# install local library
COPY DB /DB
RUN cd DB; pip install .

# change to workspace
WORKDIR /app
COPY requirements.txt /app
RUN pip install -r requirements.txt
RUN pip install torch==1.13.1+cpu torchvision==0.14.1+cpu torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cpu
