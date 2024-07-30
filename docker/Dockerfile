FROM ubuntu:focal

ENV DEBIAN_FRONTEND=noninteractive 
ENV PIP_ROOT_USER_ACTION=ignore

RUN apt-get update
RUN apt-get install -y git build-essential cmake scons python3-pip python-is-python3 gdb 
RUN apt-get install -y libboost-all-dev protobuf-compiler libconfig++-dev libyaml-cpp-dev libgpm-dev libncurses-dev libtinfo-dev

WORKDIR /setup/accelergy
RUN git clone https://github.com/HewlettPackard/cacti.git
RUN git clone https://github.com/Accelergy-Project/accelergy.git
RUN git clone https://github.com/Accelergy-Project/accelergy-aladdin-plug-in.git
RUN git clone https://github.com/Accelergy-Project/accelergy-cacti-plug-in.git
RUN git clone https://github.com/Accelergy-Project/accelergy-table-based-plug-ins.git
RUN cd cacti && make -j4
RUN cd accelergy && git reset --hard 9dc7af1789a96d50a1cad50d9d198bcad759187b && pip3 install .
RUN cd accelergy-aladdin-plug-in/ && git reset --hard 7fe410252d7aa515ed1d6dc8a29eaa2c4d5f3eaa && pip3 install .
RUN cd accelergy-cacti-plug-in/ && git reset --hard 643e6fc7635e9f15d0dbd019bc7fb7c8445e7af1 && pip3 install .
RUN cp -r cacti /usr/local/share/accelergy/estimation_plug_ins/accelergy-cacti-plug-in/
RUN cd accelergy-table-based-plug-ins/ && git reset --hard 6c5d15dac3491a485f3d8abde2d8596aa1f8221f && pip3 install .

RUN accelergy
RUN accelergyTables

WORKDIR /workspaces/
CMD export PATH=$PATH:$ADD_PATH;

RUN pip install --upgrade pip
RUN pip install hatch pandas numpy matplotlib ipykernel PyYAML seaborn pandas
RUN pip install alive-progress
