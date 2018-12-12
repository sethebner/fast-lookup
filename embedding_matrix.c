#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <assert.h>

#define NUM_WORDS 2000  // number of rows
#define EMBED_DIM 100   // number of columns
#define MAX_VALUE 100   // maximum value of an embedding dimension
#define NUM_QUERIES 100  // number of word id queries
#define ITERATIONS 100000  // number of timing loops

#define QUERY_FILE "real_queries.txt"

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

void print2d(float arr[NUM_QUERIES][EMBED_DIM])
{
  int i,j;
  for(i = 0; i < NUM_QUERIES; i++)
  {
    for(j = 0; j < EMBED_DIM; j++)
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

  int queries[NUM_QUERIES];

  float responses[NUM_QUERIES][EMBED_DIM];
  /*
  float** responses;
  float* temp2;
  responses = malloc(NUM_QUERIES * sizeof(float*));
  temp2 = malloc(NUM_QUERIES * EMBED_DIM * sizeof(float));
  assert(responses);
  assert(temp2);
  for (int i = 0; i < NUM_QUERIES; i++)
  {
    responses[i] = temp2 + (i * EMBED_DIM);
  }
  */

  // Create a sequence of queries.
  FILE *fp;
  int value;
  int idx = -1;
  fp = fopen(QUERY_FILE, "r");
  assert(fp);
  while (!feof(fp) && fscanf(fp, "%d", &value) && idx++ < NUM_QUERIES)
  {
    queries[idx] = value;
  }
  fclose(fp);


  int j, s, t;
  clock_t start, end, total_time;
  double cpu_time_used;
  float* row;
  int total_queries = 0;

  for(s = 0; s < ITERATIONS; s++)
  {
    // Putting the timing here prevents compiler optimization from running only the last iteration.
    start = clock();
    for(t = 0; t < NUM_QUERIES; t++)
    {
      int query = queries[t];
      row = matrix[query];
      for(j = 0; j < EMBED_DIM; j++)
      {
        //row[j] = matrix[query][j];
        responses[t][j] = row[j];
        total_queries++;
        //responses[t][j] = matrix[query][j];
        //responses[t][j] = matrix[queries[t]][j];
      }
    }
    end = clock();
    total_time += end - start;
    //printf("[it %d] %lu sec\n", s, (end - start)/CLOCKS_PER_SEC);
  }

  cpu_time_used = ((double) (total_time)) / CLOCKS_PER_SEC;

  printf("%f sec (total, %d iter, %d queries/iter)\n", cpu_time_used, ITERATIONS, NUM_QUERIES);
  printf("%f sec (per iter, %d queries/iter)\n", cpu_time_used/ITERATIONS, NUM_QUERIES);
  printf("Completed %d queries\n", NUM_QUERIES*ITERATIONS);

  assert(total_queries==(ITERATIONS*NUM_QUERIES*EMBED_DIM));
  printf("Total floats queried=%d\n", total_queries);
  // Clean up
  free(temp);
  free(matrix);

  /*
  free(temp2);
  free(responses);
  */
}
