FLAGS= -DDEBUG
LIBS= -lm
ALWAYS_REBUILD=makefile

nbody: nbody.o computeCUDA.o
	nvcc $(FLAGS) $^ -o $@ $(LIBS)

nbody.o: nbody.c planets.h config.h vector.h compute.h $(ALWAYS_REBUILD)
	nvcc $(FLAGS) -c $< 

computeCUDA.o: computeCUDA.cu config.h vector.h compute.h $(ALWAYS_REBUILD)
	nvcc $(FLAGS) -c $< 

clean:
	rm -f *.o nbody
