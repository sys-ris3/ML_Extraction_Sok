CC=aarch64-linux-gnu-gcc
CFLAGS=-I. -O2
LDFLAGS=-lm
DEPS = eval.h sqrt.h
OBJ = eval.o sqrtf.o ta_sqrt.o neon_sqrt.o ieee754_sqrt.o

%.o: %.c %(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS)
eval: $(OBJ)
	$(CC) -o $@ $^ $(LDFLAGS) 

clean:
	rm *.o eval

