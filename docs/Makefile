all: html

html:
	sphinx-build -b html source build/html -W --keep-going

linkcheck:
	sphinx-build -b linkcheck source build/linkcheck

clean:
	rm -rf build/

.PHONY: all html linkcheck clean
