MACHINE_TYPE=`uname -m`
if [ ${MACHINE_TYPE} == 'x86_64' ]; then
bash dist/Anaconda2-4.0.0-Linux-x86_64.sh -b -p ./anaconda2
./anaconda2/bin/conda install --use-local dist/linux-64-theano-0.8.0-py27_0.tar.bz2
./anaconda2/bin/conda install --use-local dist/linux-64-lasagne-0.2.dev1.toli-py27_0.tar.bz2
else
bash dist/Anaconda2-4.0.0-Linux-x86.sh -b -p ./anaconda2
./anaconda2/bin/conda install --use-local dist/linux-32-theano-0.8-py27_0.tar.bz2
./anaconda2/bin/conda install --use-local dist/linux-32-lasagne-0.1-py27_0.tar.bz2
fi
