#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <assert.h>

#define NUM_WORDS 2000  // number of rows
#define EMBED_DIM 100   // number of columns
#define MAX_VALUE 100   // maximum value of an embedding dimension
#define NUM_QUERIES 1000  // number of word id queries
#define ITERATIONS 10000  // number of timing loops

void random_fill(float** arr, int m, int n)
{
  int i,j;
  for(i = 0; i < m; i++)
  {
    for(j = 0; j < n; j++)
    {
      arr[i][j] = (2.0f * ((float)rand() / (float)RAND_MAX) - 1.0f) * MAX_VALUE;
    }
  }
}

void print(float** arr, int m, int n)
{
  int i,j;
  for(i = 0; i < m; i++)
  {
    for(j = 0; j < n; j++)
    {
      printf("%f ", arr[i][j]);
    }
    printf("\n");
  }
}

int main()
{
  float f;
  printf("Size of float: %ld bytes\n",sizeof(f));

  srand((unsigned int)1066);

  float** matrix;
  float* temp;

  matrix = malloc(NUM_WORDS * sizeof(float*));
  temp = malloc(NUM_WORDS * EMBED_DIM * sizeof(float));
  assert(matrix);
  assert(temp);
  for (int i = 0; i < NUM_WORDS; i++)
  {
    // enforce contiguous memory
    matrix[i] = temp + (i * EMBED_DIM);
  }

  random_fill(matrix, NUM_WORDS, EMBED_DIM);

  int queries[ITERATIONS];

  float responses[ITERATIONS][EMBED_DIM];

  // create a sequence of queries
  for(int i = 0; i < NUM_QUERIES; i++)
  {
    queries[i] = (rand() % (NUM_WORDS)) + 0;
  }

  int j, s, t;
  clock_t start, end;
  double cpu_time_used;
  float* row;

  start = clock();
  for(s = 0; s < ITERATIONS; s++)
  {
    for(t = 0; t < NUM_QUERIES; t++)
    {
      int query = queries[t];
      row = matrix[query];
      for(j = 0; j < EMBED_DIM; j++)
      {
        //row[j] = matrix[query][j];
        responses[t][j] = row[j];
        //responses[t][j] = matrix[query][j];
        //responses[t][j] = matrix[queries[t]][j];
      }
    }
  }
  end = clock();

  cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;

  printf("%f sec (total, %d iter, %d queries/iter)\n", cpu_time_used, ITERATIONS, NUM_QUERIES);
  printf("%f sec (per iter, %d queries/iter)\n", cpu_time_used/ITERATIONS, NUM_QUERIES);
  printf("Completed %d queries\n", NUM_QUERIES*ITERATIONS);

  // Clean up
  free(temp);
  free(matrix);
}
