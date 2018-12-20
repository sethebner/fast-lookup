// Adapted from https://github.com/majek/dump/blob/master/msqueue/main.c

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <pthread.h>
#include <sched.h>
#include <errno.h>
#include <unistd.h>
#include <stdint.h>
#include <math.h>

#ifdef __MACH__
#  include <mach/mach_time.h>
#endif

#include "queue.h"
#include "stddev.h"

#ifdef __MACH__
unsigned long long gettime()
{
  return mach_absolute_time();
}
#else
unsigned long long gettime()
{
  struct timespec t;
  clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &t);
  return (long long)t.tv_sec * 1000000000LL + t.tv_nsec;
}
#endif

#define NUM_PARTITIONS 2  // number of "cold" partitions

#define HOTCOLD_MODE 1
#define HOTCOLD_AWARE 1

#define NUM_WORKERS 32  // 1 per CPU

#define COLD_QUEUE_LOCKING (!((NUM_WORKERS == NUM_PARTITIONS) || (NUM_WORKERS == 1)))
#define HOT_QUEUE_LOCKING (!(NUM_WORKERS == 1))

#define WORKLOAD_SIZE 100  // number of items dequeued per queue access
#define EMPTY_POPS_BEFORE_QUIT 10

#define PARTITION_STRATEGY 1
#define CONTIGUOUS 0
#define INTERLEAVED 1

#define QUERY_STRATEGY 0
#define UNIQUE_QUERIES 1
#define REDUNDANT_QUERIES 0

#define ITERATIONS 10
#define WARMUP 3

#define MATRIX_FILE "./glove.6B/glove.6B.100d_embs.txt"
#define VOCAB_SIZE 400001
#define EMB_SIZE 100

#define NUM_HOT_WORDS (1000*(HOTCOLD_MODE == HOTCOLD_AWARE))

#define NUM_HOT_QUEUES (1*(HOTCOLD_MODE == HOTCOLD_AWARE))
#define NUM_COLD_QUEUES NUM_PARTITIONS
#define NUM_QUEUES (NUM_HOT_QUEUES + NUM_COLD_QUEUES)

#define CEIL(x,y) (x/y + (x % y != 0))
#define COLD_WORDS_PER_PARTITION (CEIL((VOCAB_SIZE - NUM_HOT_WORDS), NUM_PARTITIONS))
#define ID2COLD_PARTITION_CONTIGUOUS(id) (id / COLD_WORDS_PER_PARTITION)
#define ID2COLD_PARTITION_OFFSET_CONTIGUOUS(id) (id - NUM_HOT_WORDS - (ID2COLD_PARTITION(id)*COLD_WORDS_PER_PARTITION))
#define ID2COLD_PARTITION_INTERLEAVED(id) ((id - NUM_HOT_WORDS) % NUM_PARTITIONS)
#define ID2COLD_PARTITION_OFFSET_INTERLEAVED(id) ((id - NUM_HOT_WORDS) / NUM_PARTITIONS)

#define ID2COLD_PARTITION(id) ((PARTITION_STRATEGY == CONTIGUOUS) ? ID2COLD_PARTITION_CONTIGUOUS(id) : ID2COLD_PARTITION_INTERLEAVED(id))
#define ID2COLD_PARTITION_OFFSET(id) ((PARTITION_STRATEGY == CONTIGUOUS) ? ID2COLD_PARTITION_OFFSET_CONTIGUOUS(id) : ID2COLD_PARTITION_OFFSET_INTERLEAVED(id))

#define MAX(x,y) (((x)>(y))?(x):(y))

#define QUERY_FILE "./text/mobydick_queries.txt"
#define NUM_REDUNDANT_QUERIES 232951  // token-level
#define NUM_UNIQUE_QUERIES 13976  // type-level
#define NUM_QUERIES ((QUERY_STRATEGY == REDUNDANT_QUERIES) ? NUM_REDUNDANT_QUERIES : NUM_UNIQUE_QUERIES)

#define RESPONSE_FILE "./response/mobydick_response.txt"

#define DEBUG_MODE 1
#define DEBUG_ON 0

#define MAX_N 1000
#define MIN_N 0

volatile int running_workers = 0;
pthread_mutex_t worker_counter = PTHREAD_MUTEX_INITIALIZER;

void modify_worker_count(int delta)
{
  pthread_mutex_lock(&worker_counter);
  running_workers += delta;
  pthread_mutex_unlock(&worker_counter);
}

static void *malloc_aligned(unsigned int size)
{
  void *ptr;
  int r = posix_memalign(&ptr, 256, size);
  if (r != 0)
  {
    perror("malloc error");
    abort();
  }
  memset(ptr, 0, size);
  return ptr;
}

struct worker_data
{
  pthread_t thread_id;
  int worker_id;
  struct queue_root *work_queue;
  struct queue_root *hot_queue;
  double words_ps;
  double max_words_ps;
  unsigned long long rounds;
};

static unsigned long long threshold = 1000;

static unsigned long long total_items_processed = 0;


static float hot_embedding_matrix[NUM_HOT_WORDS][EMB_SIZE];
static float cold_embedding_matrices[NUM_PARTITIONS][COLD_WORDS_PER_PARTITION][EMB_SIZE];


static struct queue_head* query_list[NUM_QUERIES];
static float response_matrix[NUM_REDUNDANT_QUERIES][EMB_SIZE];

static void *worker_loop(void *_worker_data)
{
  // TODO: sleep a little at the start to allow all workers to be created before starting?

  unsigned long long counter = 0;
  unsigned long long t0, t1;
  struct worker_data *td = (struct worker_data *)_worker_data;
  struct queue_root *work_queue = td->work_queue;
  struct queue_root *hot_queue = td->hot_queue;

  struct queue_head *item = malloc_aligned(sizeof(struct queue_head));
  INIT_QUEUE_HEAD(item, -1, NULL); // dummy initialize

  t0 = gettime();
  struct queue_head *workload[WORKLOAD_SIZE];

  int i, j;
  int retrieved = 0;
  struct row_id_list *row_head;
  int row_id;
  float* row;
  int word_id;
  int partition_id, lookup_id;
  int empty_pops = 0;

  td->max_words_ps = 0;

  while (1)
  {
    int processed = 0;
    // prioritize "cold" words over "hot" words because cold words can be accessed only by a unique worker
    retrieved = queue_get_n(work_queue, workload, WORKLOAD_SIZE, td->worker_id, COLD_QUEUE_LOCKING);

    if (retrieved == 0)
    {
      if (HOTCOLD_MODE == HOTCOLD_AWARE)
      {
        // pull word_ids off of hot queue
        retrieved = queue_get_n(hot_queue, workload, WORKLOAD_SIZE, td->worker_id, HOT_QUEUE_LOCKING);
        if (retrieved == 0)
        {
          // nothing left to do
          empty_pops++;
          if (empty_pops >= EMPTY_POPS_BEFORE_QUIT)
          {
            modify_worker_count(-1);
            return NULL;
          }
          else
          {
            continue;
          }
        }
      }
      else
      {
        // nothing left to do
        empty_pops++;
        if (empty_pops >= EMPTY_POPS_BEFORE_QUIT)
        {
          modify_worker_count(-1);
          return NULL;
        }
        else
        {
          continue;
        }
      }
    }


    // look up word_id in matrix, and store in return matrix
    for (i = 0; i < retrieved; i++)
    {
      word_id = workload[i]->word_id;
      row_head = workload[i]->row_head;

      while (row_head != NULL)
      {
        row_id = row_head->row_id;

        if (row_id == -1)
        {
          break;
        }
        // printf("[%d] word_id=%d, row_id=%d\n", td->worker_id, word_id, row_id);

        if ((HOTCOLD_MODE == HOTCOLD_AWARE) && (word_id < NUM_HOT_WORDS))
        {
          // this is a hot word
          lookup_id = word_id;

          // printf("[%d] word_id=%d, row_id=%d (hot)\n", td->worker_id, word_id, row_id);
          // if ((row_id >= NUM_REDUNDANT_QUERIES) || (row_id == word_id))
          // {
          //   printf("Conflict: %d\n", row_id);
          //   break;
          // }

          for(j = 0; j < EMB_SIZE; j++)
          {
            // response_matrix[row_id][j] = row[j];
            response_matrix[row_id][j] = hot_embedding_matrix[lookup_id][j];
          }
        }
        else
        {
          // this is a cold word
          partition_id = ID2COLD_PARTITION(word_id);
          lookup_id = ID2COLD_PARTITION_OFFSET(word_id);

          // printf("[%d] word_id=%d, row_id=%d (cold)\n", td->worker_id, word_id, row_id);
          // if ((row_id >= NUM_REDUNDANT_QUERIES) || (row_id == word_id))
          // {
          //   printf("Conflict: %d\n", row_id);
          //   break;
          // }

          for(j = 0; j < EMB_SIZE; j++)
          {
            // response_matrix[row_id][j] = row[j];
            response_matrix[row_id][j] = cold_embedding_matrices[partition_id][lookup_id][j];
          }
        }

        processed++;
        row_head = row_head->next;
      }
    }

    total_items_processed += processed;


    counter += retrieved;
    if (counter < threshold)
    {
      continue;
    }
    else
    {
      // profiling stats
      t1 = gettime();
      unsigned long long delta = t1 - t0;
      double words_ps = (double)counter / ((double) delta / 1000000000LL);
      td->words_ps = words_ps;
      td->max_words_ps = MAX(words_ps, td->max_words_ps);
      td->rounds += 1;
      // t0 = t1;

      #if DEBUG_ON
      printf("worker=%d round done: %llu in %.3fms, words per sec=%.3f\n",
             td->worker_id,
             counter,
             (double)delta / 1000000,
             words_ps);
      #endif

      counter = 0;
      t0 = gettime();
    }
  }

  modify_worker_count(-1);

  return NULL;
}

int assign_queue(int word_id)
{
  // Give each queue/partition a contiguous block of word ids (cache locality)
  if (HOTCOLD_MODE == HOTCOLD_AWARE)
  {
    if (word_id < NUM_HOT_WORDS) { return 0; } // hot queue, which is available to all nodes
    else { return ID2COLD_PARTITION(word_id) + 1; }
  }
  else { return ID2COLD_PARTITION(word_id); }
}

int get_random()
{
  return rand() % (MAX_N + 1 - MIN_N) + MIN_N;
}

int main(int argc, char **argv)
{
  int i, j, k;
  int unused;
  int prefill = NUM_QUERIES; // size of batch

  fprintf(stderr, "Running with npartitions=%i, prefill=%i\n", NUM_PARTITIONS, prefill);
  threshold /= NUM_PARTITIONS;

  if (HOTCOLD_MODE == HOTCOLD_AWARE)
  {
    printf("Hot-Cold aware. Privileged hot queue created.\n");
  }
  else
  {
    printf("NOT Hot-Cold aware. Privileged hot queue NOT created.\n");
  }

  // Create embedding matrices
  FILE *file;
  file = fopen(MATRIX_FILE, "r");
  if (file == NULL)
  {
    printf("Matrix file not found.\n");
    exit(1);
  }

  if (HOTCOLD_MODE == HOTCOLD_AWARE)
  {
    for (i = 0; i < NUM_HOT_WORDS; i++)
    {
      for (j = 0; j < EMB_SIZE; j++)
      {
        unused = fscanf(file, "%f", &(hot_embedding_matrix[i][j]));
      }
    }
  }

  if (PARTITION_STRATEGY == CONTIGUOUS)
  {
    for (k = 0; k < NUM_PARTITIONS; k++)
    {
      for (i = 0; i < COLD_WORDS_PER_PARTITION; i++)
      {
        for (j = 0; j < EMB_SIZE; j++)
        {
          unused = fscanf(file, "%f", &(cold_embedding_matrices[k][i][j]));
        }
      }
    }
  }
  else if (PARTITION_STRATEGY == INTERLEAVED)
  {
    for (i = 0; i < COLD_WORDS_PER_PARTITION; i++)
    {
      for (k = 0; k < NUM_PARTITIONS; k++)
      {
        for (j = 0; j < EMB_SIZE; j++)
        {
          unused = fscanf(file, "%f", &(cold_embedding_matrices[k][i][j]));
        }
      }
    }
  }
  else
  {
    printf("Unrecognized partitioning strategy.\n");
    exit(1);
  }
  fclose(file);

  // Create query list
  file = fopen(QUERY_FILE, "r");
  if (file == NULL)
  {
    printf("Query file not found.\n");
    exit(1);
  }

  printf("Reading in queries...\n");
  int word_id;
  if (QUERY_STRATEGY == REDUNDANT_QUERIES)
  {
    for (i = 0; i < NUM_QUERIES; i++)
    {
      unused = fscanf(file, "%d", &word_id);
      struct queue_head *item = malloc_aligned(sizeof(struct queue_head));
      struct row_id_list *row_head = malloc_aligned(sizeof(struct row_id_list));
      row_head->row_id = -1;
      row_head->next = NULL;
      push(&row_head, i);
      INIT_QUEUE_HEAD(item, word_id, row_head);
      query_list[i] = item;
    }
  }
  else if (QUERY_STRATEGY == UNIQUE_QUERIES)
  {
    char buffer[1000000];
    char *pbuff;
    int value;
    for (i = 0; i < NUM_QUERIES; i++)
    {
      if (!fgets(buffer, sizeof(buffer), file)) { break; }
      pbuff = buffer;

      struct queue_head *item = malloc_aligned(sizeof(struct queue_head));
      struct row_id_list *row_head = malloc_aligned(sizeof(struct row_id_list));
      row_head->row_id = -1;
      row_head->next = NULL;

      // read line from query file
      int seen = 0;
      word_id = -1;
      while (1)
      {
        // privileged first int is the word id, all others are row ids
        if (*pbuff == '\n') { break; }
        value = strtol(pbuff, &pbuff, 10);

        // printf("%d\n", value);
        // row_head.row_id = value;
        if (seen == 0) { word_id = value; }
        else { push(&row_head, value); }

        seen++;

      }

      INIT_QUEUE_HEAD(item, word_id, row_head);
      query_list[i] = item;
    }
  }
  else
  {
    printf("Unrecognized query strategy.\n");
    exit(1);
  }
  fclose(file);


  // Partition queues
  // holds word_id's to lookup
  struct queue_root *pqueues[NUM_QUEUES];
  for (j=0; j < NUM_QUEUES; j++)
  {
    pqueues[j] = ALLOC_QUEUE_ROOT();
  }

  struct timespec ts;
  ts.tv_sec = 1;
  ts.tv_nsec = 10;

  struct worker_data *worker_data = malloc_aligned(sizeof(struct worker_data) * NUM_WORKERS);

  printf("Starting processing...\n");
  // Create processes for each worker
  int iteration;
  int n = 0;
  float wps_avg;
  float avg_avg_throughput, avg_max_throughput, max_avg_throughput, max_max_throughput = 0;
  for (iteration=0; iteration < ITERATIONS + WARMUP; iteration++)
  {
    if (iteration < WARMUP)
    {
      printf("[W] ");
    }
    printf("Prefilling queues with queries...\n");
    for (i=0; i < prefill; i++)
    {
      // int word_id = query_list[i].word_id;
      // struct queue_head *item = malloc_aligned(sizeof(struct queue_head));
      // INIT_QUEUE_HEAD(item, word_id, i, &query_list[i].row_head);
      // printf("i=%d, word_id=%d\n", i, query_list[i]->word_id);
      int queue_id = assign_queue(query_list[i]->word_id);
      //printf("Putting word_id=%i in queue=%i\n", item->word_id, queue_id);
      queue_put(query_list[i], pqueues[queue_id]);
    }
    // printf("Queues filled.\n");
    for (i=0; i < NUM_WORKERS; i++)
    {
      modify_worker_count(1);

      // assign worker to a partition's queue and to the hot queue if it exists
      worker_data[i].worker_id = i;
      if (HOTCOLD_MODE == HOTCOLD_AWARE)
      {
        worker_data[i].work_queue = pqueues[i%NUM_COLD_QUEUES + 1];
        worker_data[i].hot_queue = pqueues[0];
        // printf("w=%d -> q=%d\n", worker_data[i].worker_id, i%num_cold_queues + 1);
      }
      else
      {
        worker_data[i].work_queue = pqueues[i % NUM_QUEUES];
        worker_data[i].hot_queue = NULL;
        // printf("w=%d -> q=%d\n", worker_data[i].worker_id, i % num_queues);
      }

      // worker_data[i].worker_id = i;
      int r = pthread_create(&worker_data[i].thread_id,
                             NULL,
                             &worker_loop,
                             &worker_data[i]);
      if (r != 0)
      {
        perror("pthread_create()");
        abort();
      }
    }

    // Collect statistics
    double avg, dev, rounds_avg, rounds_dev, words_avg, words_dev = 0;
    double throughput = 0;
    double avg_throughput = 0;
    int reports = 0;

    double max_throughput = 0;
    while (running_workers > 0)
    {
      // nanosleep(&ts, NULL);
      // Single round shorter than 1 ms?
      // if (threshold / worker_data[0].words_ps < 0.001)
      // {
      //   fprintf(stderr, "threshold %lli -> %lli\n",
      //           threshold,
      //           threshold*2);
      //   threshold *= 2;
      //   continue;
      // }

      reports++;

      struct stddev sd, rounds, words;
      memset(&sd, 0, sizeof(sd));
      memset(&rounds, 0, sizeof(rounds));
      memset(&words, 0, sizeof(words));
      for (i=0; i < NUM_WORKERS; i++)
      {
        stddev_add(&rounds, worker_data[i].rounds);
        stddev_add(&words, worker_data[i].words_ps);
      }
      // double avg, dev, rounds_avg, rounds_dev, words_avg, words_dev;
      stddev_get(&sd, NULL, &avg, &dev);
      stddev_get(&rounds, NULL, &rounds_avg, &rounds_dev);
      stddev_get(&words, NULL, &words_avg, &words_dev);
      throughput = (float)words.sum;
      max_throughput = MAX(throughput, max_throughput);
      // printf("%.3f, %.3f, %.3f, %.3f, %.3f, %llu\n", sd.sum, avg, dev, rounds_avg, rounds_dev, threshold);
      // printf("wps: avg=%.3f, std=%.3f\n", words_avg, words_dev);
      // printf("throughput=%.3f\n", throughput);

      avg_throughput = avg_throughput*(((double)(reports - 1))/reports) + (throughput / reports);
    }

    // for (i = 0; i < NUM_WORKERS; i++)
    // {
    //   pthread_join(worker_data[i].thread_id, NULL);
    // }
    //
    // struct stddev max_words;
    // memset(&max_words, 0, sizeof(max_words));
    // for (i = 0; i < NUM_WORKERS; i++)
    // {
    //   stddev_add(&max_words, worker_data[i].max_words_ps);
    // }
    // double max_words_avg, max_words_dev;
    // stddev_get(&max_words, NULL, &max_words_avg, &max_words_dev);
    // printf("max wps: avg=%.3f, std=%.3f\n", max_words_avg, max_words_dev);

    printf("avg throughput=%.3f\n", avg_throughput);
    printf("max throughput=%.3f\n", max_throughput);
    if (iteration >= WARMUP)
    {
      n++;
      // max_wps_avg += max_words_avg;
      // wps_avg += words_avg;
      // avg_throughput += throughput;
      avg_avg_throughput = avg_avg_throughput*(((double)(n - 1))/n) + (avg_throughput / n);
      // avg_max_throughput += max_throughput;
      avg_max_throughput = avg_max_throughput*(((double)(n - 1))/n) + (max_throughput / n);
      max_max_throughput = MAX(max_throughput, max_max_throughput);
      max_avg_throughput = MAX(avg_throughput, max_avg_throughput);
    }
  }
  // max_wps_avg /= ITERATIONS;
  // wps_avg /= ITERATIONS;
  // avg_throughput /= ITERATIONS;
  // avg_max_throughput /= ITERATIONS;

  // Write out response matrix
  /*
  printf("Writing out responses to file...\n");
  file = fopen(RESPONSE_FILE, "w");
  if (file == NULL)
  {
    printf("Response file not found.\n");
    exit(1);
  }

  for (i = 0; i < NUM_QUERIES; i++)
  {
    fprintf(file, "[%d]: ", query_list[i]);
    for (j = 0; j < EMB_SIZE; j++)
    {
      fprintf(file, "%f ", response_matrix[i][j]);
    }
    fprintf(file, "\n");
  }
  fclose(file);
  */

  // printf("avg wps: %f\n", wps_avg);
  printf("avg avg throughput=%.3f\n", avg_avg_throughput);
  printf("avg max throughput=%.3f\n", avg_max_throughput);
  printf("max avg throughput=%.3f\n", max_avg_throughput);
  printf("max max throughput=%.3f\n", max_max_throughput);
  printf("%llu items processed, %d items assigned\n", total_items_processed, NUM_REDUNDANT_QUERIES*(ITERATIONS+WARMUP));

  return 0;
}
