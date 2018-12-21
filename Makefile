CC=gcc
#OPT=-O0
OPT=-O3

dispatch: queue_dispatch.c
	$(CC) $(OPT) -pthread queue_dispatch.c queue.c -o dispatch -lm
