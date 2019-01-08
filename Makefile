THXX_ROOT := $(PWD)
include conf.mk

.PHONY: test clean install-conda install-latest third_party doc

test: third_party
	$(MAKE) --directory test

example-mnist: third_party
	$(MAKE) --directory example/mnist

example-copy: third_party
	$(MAKE) --directory example/copy

example-speech-recognition: third_party
	$(MAKE) --directory example/speech_recognition


clean:
	find . -name "*.o" -exec rm -v {} \;
	find . -name "*.out" -exec rm -v {} \;

third_party:
	$(MAKE) --directory third_party

doc:
	$(MAKE) --directory doc

