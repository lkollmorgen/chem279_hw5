# make the environment
env:
	conda create -n hw4 "python=3.11" matplotlib conda-forge::armadillo conda-forge::eigen

#do everything
everything:
	cd src; make -s all
	cd test; make -s all

