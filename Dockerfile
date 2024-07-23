############### Build image ##################
FROM ubuntu:18.04
FROM python:3.12

# using this label we can clean up current cached images
LABEL delete_when_outdate=yes

    # python
ENV PYTHONUNBUFFERED=1
    # prevents python creating .pyc files
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONFAULTHANDLER=1
ENV PYTHONHASHSEED=random

    # pip
ENV PIP_NO_CACHE_DIR=off
ENV PIP_DISABLE_PIP_VERSION_CHECK=on
ENV PIP_DEFAULT_TIMEOUT=100



ENV CLEARML_WEB_SERVER=https://app.clear.ml/
ENV CLEARML_API_SERVER=https://api.clear.ml
ENV CLEARML_FILES_SERVER=https://files.clear.ml
ENV CLEARML_ACCESS_KEY=GDHQKX48TLCWO6MPAPVC24QFSKWWOP
ENV CLEARML_SECRET_KEY=4Hr7VZaEW-DG8CKCLpNtMjD35PgPAswqyqQCdoY6N9iwO2xQGCqHxrE-vsxJU_BxiJQ



# Install pipenv and compilation dependencies
# TODO: Make "code" directory via docker-compose file
WORKDIR /code/backend/
# TODO: Зачем два раза копировать? Сначала Pipfile, потом code/backend
COPY configs /code/backend/
#COPY requirenments/requirenments-gpu.txt /code/backend/
COPY requirenments/requirenments-base.txt /code/backend/
COPY src /code/backend
COPY Makefile /code/backend
COPY Dockerfile /code/backend
COPY clearml_init.sh /code/backend
RUN python -m pip install --upgrade pip
RUN pip install -r /code/backend/requirenments-base.txt
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
# Code

# Make the script executable
RUN chmod +x /code/backend/clearml_init.sh

# Run the initialization script
CMD ["/code/backend/clearml_init.sh"]