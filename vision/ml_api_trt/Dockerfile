FROM nvcr.io/nvidia/l4t-tensorrt:r10.3.0-devel

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
	build-essential gcc g++ \
	python3 python3-pip python3-dev git curl \
	libglib2.0-0 libsm6 libxext6 libxrender1 \
	python3-opencv \
	&& rm -rf /var/lib/apt/lists/*

COPY . /app

WORKDIR /app

COPY model /app/model
COPY lib /app/lib

RUN pip install -r requirements.txt

# make sure CUDA is visible
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=$CUDA_HOME/bin:$PATH
ENV LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

RUN pip3 install --no-cache-dir pycuda

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh



EXPOSE 3333

ENTRYPOINT ["/entrypoint.sh"]


CMD ["python3", "manage.py", "runserver", "0.0.0.0:3335"]