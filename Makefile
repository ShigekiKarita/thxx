.PHONY: test clean

test:
	$(MAKE) --directory test

clean:
	find . -name "*.o" -exec rm -v {} \;
	find . -name "*.out" -exec rm -v {} \;
