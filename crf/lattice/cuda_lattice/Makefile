bilateral: Image.cpp Image.h bilateral.cpp jpg_file.h png_file.h macros.h permutohedral.o
	nvcc -arch=sm_61 -O3 bilateral.cpp Image.cpp permutohedral.o -o bilateral  -lpng -ljpeg -L. -lcutil

permutohedral.o: permutohedral.cu hash_table.cu
	nvcc -use_fast_math -O3 -DLIBRARY -c permutohedral.cu -o permutohedral.o

clean: 
	rm bilateral permutohedral.o