FROM nvidia/cuda:12.3.1-devel-ubuntu20.04


ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV PATH /opt/conda/bin:$PATH

# Install Dependencies of Miniconda
RUN apt-get update --fix-missing && \
    apt-get install -y wget bzip2 curl git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install miniconda3
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda acppptivate CUBIG" >> ~/.bashrc


# Copy the environment.yaml file to the container
COPY environment.yml .

RUN /opt/conda/bin/conda env create -f environment.yml && \
    /opt/conda/bin/conda clean -afy


# SHELL ["conda", "run", "-n", "CUBIG", "/bin/bash", "-c"]

RUN apt-get update &&\
    DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt-get -y install tzdata

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

RUN mkdir .cache

# Copy the rest of the project files to the container
COPY . .

# Set the working directory
WORKDIR .

# Set the environment variable PATH to include Conda binaries
# ENV PATH="/opt/conda/envs/CUBIG/bin:$PATH"

RUN apt-get update

# Install MariaDB Connector/C
RUN apt-get install -y curl gcc

# RUN curl -sS https://downloads.mariadb.com/MariaDB/mariadb_repo_setup | bash

ENV PATH="/opt/conda/envs/CUBIG/bin:$PATH"

RUN chmod 1777 /tmp

RUN curl -sS https://downloads.mariadb.com/MariaDB/mariadb_repo_setup | bash

RUN apt-get install -y libmariadb3 libmariadb-dev

RUN apt-get autoremove libmysqlclient-dev

RUN export CFLAGS="-std=c99"

# RUN mkdir /tmp/connector
# RUN wget https://downloads.mariadb.com/Connectors/c/connector-c-3.3.5/mariadb-connector-c-3.3.5-ubuntu-bionic-amd64.tar.gz --directory /tmp/connector

# # -----------------------------------------------------------------------------------------
# # use tar to extract the MariaDB Connector that you downloaded into the temporary directory
# # -----------------------------------------------------------------------------------------
# RUN tar -zxpf /tmp/connector/mariadb-connector-c-3.3.5-ubuntu-bionic-amd64.tar.gz --directory /tmp/connector

# # --------------------------------
# # move files
# # --------------------------------
# RUN mv -f /tmp/connector/mariadb-connector-c-3.3.5-ubuntu-bionic-amd64/bin/mariadb_config /usr/bin/
# RUN mv -f /tmp/connector/mariadb-connector-c-3.3.5-ubuntu-bionic-amd64/include/mariadb /usr/local/include/
# RUN mv -f /tmp/connector/mariadb-connector-c-3.3.5-ubuntu-bionic-amd64/lib/mariadb /usr/local/lib/

RUN /bin/bash -c "source activate CUBIG && pip3 install mariadb" 


CMD [ "/bin/bash" ]