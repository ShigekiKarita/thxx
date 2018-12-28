all: pytorch_sphinx_theme
	doxygen
	# pip install breathe
	cd sphinx && make html

pytorch_sphinx_theme:
	git clone https://github.com/ShigekiKarita/thxx_sphinx_theme --depth 1
	ln -s $(PWD)/thxx_sphinx_theme pytorch_sphinx_theme
