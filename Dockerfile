# Use specific base images
FROM ubuntu:18.04
FROM python:3.12

# Using this label we can clean up current cached images
LABEL delete_when_outdate=yes

# Set python environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONFAULTHANDLER=1
ENV PYTHONHASHSEED=random

ENV CLEARML_WEB_SERVER=https://app.clear.ml/
ENV CLEARML_API_SERVER=https://api.clear.ml
ENV CLEARML_FILES_SERVER=https://files.clear.ml
ENV CLEARML_ACCESS_KEY=GDHQKX48TLCWO6MPAPVC24QFSKWWOP
ENV CLEARML_SECRET_KEY=4Hr7VZaEW-DG8CKCLpNtMjD35PgPAswqyqQCdoY6N9iwO2xQGCqHxrE-vsxJU_BxiJQ
ENV CLEARML_API_SECRET_KEY=4Hr7VZaEW-DG8CKCLpNtMjD35PgPAswqyqQCdoY6N9iwO2xQGCqHxrE-vsxJU_BxiJQ
ENV CLEARML_API_ACCESS_KEY=GDHQKX48TLCWO6MPAPVC24QFSKWWOP

# Set pip environment variables
ENV PIP_NO_CACHE_DIR=off
ENV PIP_DISABLE_PIP_VERSION_CHECK=on
ENV PIP_DEFAULT_TIMEOUT=100

# Copy the .env file and source it
COPY .env /code/backend/.env

# Set working directory
WORKDIR /code/backend/

# Copy necessary files
COPY configs /code/backend/
COPY requirenments/requirenments-base.txt /code/backend/
COPY src /code/backend
COPY Makefile /code/backend
COPY Dockerfile /code/backend
COPY clearml_init.sh /code/backend

# Install dependencies
RUN python -m pip install --upgrade pip
RUN pip install -r /code/backend/requirenments-base.txt
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install ClearML
RUN pip install clearml
RUN chmod +x /code/backend/clearml_init.sh
# Run ClearML initialization script
#CMD ["bash", "-c", "source /code/backend/.env && /code/backend/clearml_init.sh"]
CMD ["clearml-init"]
