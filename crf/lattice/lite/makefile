appname := test
CXX := gcc
CXXFLAGS := -std=c++11

srcfiles := $(shell find . -name "*.c")
objects  := $(patsubst %.c, %.o, $(srcfiles))

all: $(appname)

$(appname): $(objects) 
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $(appname) $(objects) $(LDLIBS)

depend: .depend

.depend: $(srcfiles) 
	rm -f ./.depend 
	$(CXX) $(CXXFLAGS) -MM $^>>./.depend;

clean: 
	rm -f $(objects)

dist-clean: 
	clean rm -f *~ .depend

include .depend