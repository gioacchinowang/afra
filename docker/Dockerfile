FROM ubuntu:latest

RUN apt-get update
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y --fix-missing build-essential cmake
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y --fix-missing git wget unzip vim
RUN apt-get install -y --fix-missing python3 python3-pip python3-dev
RUN pip3 install jupyter numpy scipy astropy healpy matplotlib progressbar seaborn pytest nbval corner

RUN mkdir /home/lab
WORKDIR /home/lab

#mpi4py
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y --fix-missing libopenmpi-dev openmpi-bin openmpi-doc
RUN pip3 install mpi4py

#NaMaster
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y --fix-missing automake autoconf gsl-bin libgsl-dev

#MultiNest
RUN apt-get install -y --fix-missing libblas3 libblas-dev liblapack3 liblapack-dev libatlas-base-dev libatlas3-base
RUN git clone https://github.com/JohannesBuchner/MultiNest.git
WORKDIR /home/lab/MultiNest/build
RUN cmake -DCMAKE_INSTALL_PREFIX:PATH=/usr/local .. && make && make install
WORKDIR /home/lab

# PyMultiNest
RUN git clone https://github.com/JohannesBuchner/PyMultiNest.git
WORKDIR /home/lab/PyMultiNest
RUN python3 setup.py install
WORKDIR /home/lab

# NaMaster
RUN mkdir /home/lab/fftw
RUN wget http://www.fftw.org/fftw-3.3.8.tar.gz && tar xzf fftw-3.3.8.tar.gz -C fftw --strip-components 1
WORKDIR fftw
RUN ./configure --enable-threads --enable-openmp --enable-shared --prefix=/usr/local/ && make && make install
WORKDIR /home/lab

RUN mkdir /home/lab/cfitsio
RUN wget http://heasarc.gsfc.nasa.gov/FTP/software/fitsio/c/cfitsio_latest.tar.gz && tar xzf cfitsio_latest.tar.gz -C cfitsio --strip-components 1
WORKDIR /home/lab/cfitsio
RUN ./configure --prefix=/usr/local/ && make && make install
WORKDIR /home/lab

RUN mkdir /home/lab/healpix
RUN wget https://sourceforge.net/projects/healpix/files/Healpix_3.60/Healpix_3.60_2019Dec18.tar.gz && tar xzf Healpix_3.60_2019Dec18.tar.gz -C healpix --strip-components 1
WORKDIR /home/lab/healpix
RUN ./configure --auto=cxx && make
ENV LD_LIBRARY_PATH /usr/local/lib:/home/lab/healpix/lib:${LD_LIBRARY_PATH}
WORKDIR /home/lab

RUN git clone https://github.com/LSSTDESC/NaMaster.git namaster
WORKDIR /home/lab/namaster
RUN pip3 install .
WORKDIR /home/lab

#pysm & camb
RUN pip3 install camb pysm3

#ABSpy
RUN git clone https://github.com/gioacchinowang/ABSpy.git abspy
WORKDIR /home/lab/abspy
RUN pip3 install .
WORKDIR /home/lab

#clean up
RUN rm *.tar.gz

# alias
RUN echo "alias mpirun='mpirun --mca btl ^vader --allow-run-as-root'" >> ~/.bashrc
