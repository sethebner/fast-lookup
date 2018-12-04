CC=gcc
#OPT=-O0
OPT=-O3

embedding_matrix: embedding_matrix.c
	$(CC) $(OPT) -o embedding_matrix embedding_matrix.c
