FROM nvidia/cuda:10.0-devel-ubuntu18.04

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV PATH /opt/conda/bin:$PATH

# Install dependencies
RUN apt-get update --fix-missing && apt-get install -y --no-install-recommends \
    wget \
    bzip2 \
    ca-certificates \
    curl \ 
    git \
    curl \
    cmake \
    build-essential \
    ninja-build \
    libatlas-base-dev \
    libflann-dev \
    libboost1.65-all-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Download Eigen
RUN curl -SL http://bitbucket.org/eigen/eigen/get/default.tar.gz \
    | tar -xzC /tmp \
    && mv /tmp/eigen-eigen-ea671884cc96 /tmp/eigen 

# Download PCL
RUN curl -SL https://github.com/PointCloudLibrary/pcl/archive/pcl-1.9.0.tar.gz \
    | tar -xzC /tmp \
    && mv /tmp/pcl-pcl-1.9.0 /tmp/pcl 

# Build and install Eigen
RUN mkdir -p /tmp/eigen/build \
    && cd /tmp/eigen/build \
    && cmake -GNinja .. \
    && ninja \
    && ninja install 

# Build and install PCL
RUN mkdir -p /tmp/pcl/build \
    && cd /tmp/pcl/build \
    && cmake -GNinja  \
    -DCMAKE_BUILD_TYPE=Release \
    -DWITH_VTK=OFF \
    -DWITH_PCAP=OFF \
    -DWITH_LIBUSB=OFF \
    -DWITH_OPENGL=OFF \
    -DBUILD_2d=ON \
    -DBUILD_CUDA=OFF \
    -DBUILD_GPU=OFF \
    -DBUILD_apps=OFF \
    -DBUILD_common=ON \
    -DBUILD_examples=OFF \
    -DBUILD_features=ON \
    -DBUILD_filters=ON \
    -DBUILD_geometry=OFF \
    -DBUILD_global_tests=OFF \
    -DBUILD_io=ON \
    -DBUILD_kdtree=ON  \
    -DBUILD_keypoints=OFF \
    -DBUILD_ml=OFF \
    -DBUILD_octree=ON \
    -DBUILD_outofcore=OFF \
    -DBUILD_people=OFF \
    -DBUILD_recognition=OFF \
    -DBUILD_registration=ON \
    -DBUILD_sample_consensus=ON \
    -DBUILD_search=ON \
    -DBUILD_segmentation=OFF \
    -DBUILD_simulation=OFF \
    -DBUILD_stereo=OFF \
    -DBUILD_surface=OFF  \
    -DBUILD_tools=OFF \
    -DBUILD_tracking=OFF \
    -DBUILD_visualization=OFF \
    .. \
    && ninja -j32 \
    && ninja install

# Install Miniconda3
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-4.5.11-Linux-x86_64.sh -O ~/miniconda.sh \
    && /bin/bash ~/miniconda.sh -b -p /opt/conda \
    && rm ~/miniconda.sh \
    && /opt/conda/bin/conda clean -tipsy \
    && ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh \
    && echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc \
    && echo "conda activate base" >> ~/.bashrc

RUN conda install -c conda-forge -y \
    numpy \
    matplotlib \
    notebook \
    pythreejs \
    pybind11

RUN pip install pycollada https://github.com/tingelst/tpk4170-robotics/archive/master.zip

ENV TINI_VERSION v0.16.1
ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /usr/bin/tini
RUN chmod +x /usr/bin/tini

COPY run_jupyter.sh /

# Jupyter notebook
EXPOSE 8888

ENTRYPOINT [ "/usr/bin/tini", "--" ]
CMD [ "/bin/bash" ]