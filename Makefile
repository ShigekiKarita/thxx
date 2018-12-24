.PHONY: test clean install-conda install-latest

test:
	$(MAKE) --directory test

clean:
	find . -name "*.o" -exec rm -v {} \;
	find . -name "*.out" -exec rm -v {} \;
