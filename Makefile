CC=gcc
#OPT=-O0
OPT=-O3

MPICC?=/Users/sethebner/opt/usr/local/bin/mpicc

embedding_matrix: embedding_matrix.c
	$(CC) $(OPT) -o embedding_matrix embedding_matrix.c

dispatch_demo: queue_dispatch_demo.c
	$(CC) $(OPT) queue_dispatch_demo.c queue.c -o dispatch_demo

dispatch: queue_dispatch.c
	$(CC) $(OPT) queue_dispatch.c queue.c -o dispatch

mpi_hello_world: mpi_hello_world.c
	${MPICC} -o mpi_hello_world mpi_hello_world.c
