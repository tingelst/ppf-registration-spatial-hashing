pushd build
cmake -DPYTHON_EXECUTABLE=/home/lars/bin/miniconda3/envs/tpk4170/bin/python ..
make -j64
popd