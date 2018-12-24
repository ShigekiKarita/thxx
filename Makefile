.PHONY: test clean install-conda install-latest

test:
	$(MAKE) --directory test

clean:
	find . -name "*.o" -exec rm -v {} \;
	find . -name "*.out" -exec rm -v {} \;

install-conda:
  wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh;
  bash miniconda.sh -b -p $HOME/miniconda
	source $HOME/miniconda/bin/activate; echo $CONDA_PREFIX
  source $HOME/miniconda/bin/activate; hash -r
  source $HOME/miniconda/bin/activate; conda config --set always_yes yes --set changeps1 no
  source $HOME/miniconda/bin/activate; conda update -q conda
  source $HOME/miniconda/bin/activate; conda info -a
  source $HOME/miniconda/bin/activate; conda install pytorch-cpu -c pytorch

install-latest:
	wget https://download.pytorch.org/libtorch/cpu/libtorch-shared-with-deps-latest.zip
	unzip libtorch-shared-with-deps-latest.zip

