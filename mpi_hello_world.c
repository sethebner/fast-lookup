#include <mpi.h>
#include <stdio.h>

int main(int argc, char** argv)
{
  // initialize MPI environment
  MPI_Init(&argc, &argv);

  // get number of processes
  int world_size;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  // get rank of process
  int world_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  //get name of processor
  char processor_name[MPI_MAX_PROCESSOR_NAME];
  int name_len;
  MPI_Get_processor_name(processor_name, &name_len);

  printf("Hello world from processor %s, rank %d of %d processors\n", processor_name, world_rank, world_size);

  // finalize MPI environment
  MPI_Finalize();

  return 0;
}
