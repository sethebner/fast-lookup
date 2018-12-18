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

#define NUM_PARTITIONS 1  // number of "cold" partitions

#define HOTCOLD_MODE 1
#define HOTCOLD_AWARE 0

#define NUM_WORKERS 1  // 1 per CPU
// #define WORKERS_PER_PARTITION 2 // !!

#define WORKLOAD_SIZE 100  // number of items dequeued per queue access

#define ITERATIONS 10
#define WARMUP 3

#define MAX_N 1000
#define MIN_N 0

#define DEBUG_MODE 1
#define DEBUG_ON 0

#define MATRIX_FILE "./glove.6B/glove.6B.100d_embs.txt"
#define VOCAB_SIZE 400001
#define EMB_SIZE 100
#define NUM_HOT_WORDS 1000

#define QUERY_FILE "./text/mobydick_queries.txt"
#define NUM_QUERIES 232951

#define RESPONSE_FILE "./response/mobydick_response.txt"

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
  unsigned long long rounds;
};

static unsigned long long threshold = 1000;

static unsigned long long total_items_processed = 0;

static float embedding_matrix[VOCAB_SIZE][EMB_SIZE];
static int query_list[NUM_QUERIES];
static float response_matrix[NUM_QUERIES][EMB_SIZE];

static void *worker_loop(void *_worker_data)
{
  // TODO: sleep a little at the start to allow all workers to be created before starting?

  unsigned long long counter = 0;
  unsigned long long t0, t1;
  struct worker_data *td = (struct worker_data *)_worker_data;
  struct queue_root *work_queue = td->work_queue;
  struct queue_root *hot_queue = td->hot_queue;

  struct queue_head *item = malloc_aligned(sizeof(struct queue_head));
  INIT_QUEUE_HEAD(item, -1, -1); // dummy initialize

  t0 = gettime();
  struct queue_head *workload[WORKLOAD_SIZE];

  int i, j;
  int retrieved = 0;
  int row_id;
  float* row;
  int word_id;

  while (1)
  {
    // prioritize "cold" words over "hot" words because cold words can be accessed only by a unique worker
    retrieved = queue_get_n(work_queue, workload, WORKLOAD_SIZE, td->worker_id);

    if (retrieved == 0)
    {
      if (HOTCOLD_MODE == HOTCOLD_AWARE)
      {
        // pull word_ids off of hot queue
        retrieved = queue_get_n(hot_queue, workload, WORKLOAD_SIZE, td->worker_id);
        if (retrieved == 0)
        {
          // nothing left to do
          modify_worker_count(-1);
          return NULL;
        }
        else
        {
        }
      }
      else
      {
        // nothing left to do
        modify_worker_count(-1);

        return NULL;
      }
    }
    else
    {
    }


    // look up word_id in matrix, and store in return matrix
    for (i = 0; i < retrieved; i++)
    {
      row_id = workload[i]->row_id;
      word_id = workload[i]->word_id;
      // row = embedding_matrix[word_id];
      // memcpy(response_matrix[row_id], (float*)row, sizeof response_matrix[row_id]); // slow
      for(j = 0; j < EMB_SIZE; j++)
      {
        // response_matrix[row_id][j] = row[j];
        response_matrix[row_id][j] = embedding_matrix[word_id][j];
      }
    }

    total_items_processed += retrieved;


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

int assign_queue(int word_id, int num_cold_queues, int num_hot_words)
{
  if (HOTCOLD_MODE == HOTCOLD_AWARE)
  {
    if (word_id < num_hot_words) { return 0; } // hot queue available to all nodes
    else { return word_id % num_cold_queues + 1; } // do not assign to hot queue; assign to appropriate cold queue
  }
  else { return word_id % num_cold_queues; } // no hot queue, so assign to a cold queue; ignores #items in queues/load balancing
}

int get_random()
{
  return rand() % (MAX_N + 1 - MIN_N) + MIN_N;
}

int main(int argc, char **argv)
{
  int i, j;
  // int num_partitions = 1; // number of partitions
  int prefill = NUM_QUERIES; // size of batch
  int num_cold_queues, num_hot_queues, num_queues, num_workers;


  if (argc > 1)
  {
    // num_partitions = atoi(argv[1]);  // number of cold partitions
  }
  if (argc > 2)
  {
    // prefill = atoi(argv[2]);
  }
  if (argc > 3)
  {
    fprintf(stderr, "Usage: %s [npartitions] [prefill]\n", argv[0]);
    abort();
  }
  fprintf(stderr, "Running with npartitions=%i, prefill=%i\n", NUM_PARTITIONS, prefill);
  threshold /= NUM_PARTITIONS;

  num_cold_queues = NUM_PARTITIONS;

  if (HOTCOLD_MODE == HOTCOLD_AWARE)
  {
    num_hot_queues = 1;
    // num_queues = num_partitions + 1; // queue 0 is a privileged queue for "hot" words that all nodes may access
  }
  else
  {
    num_hot_queues = 0;
    // num_queues = num_partitions;
  }
  num_queues = num_cold_queues + num_hot_queues;

  // TODO: check this for correctness. should we have num_workers = num_nodes * WORKERS_PER_NODE?
  // num_workers = num_partitions * WORKERS_PER_PARTITION;
  num_workers = NUM_WORKERS;

  if (HOTCOLD_MODE == HOTCOLD_AWARE)
  {
    printf("Hot-Cold aware. Privileged hot queue created.\n");
  }
  else
  {
    printf("NOT Hot-Cold aware. Privileged hot queue NOT created.\n");
  }

  // Create embedding matrix
  FILE *file;
  file = fopen(MATRIX_FILE, "r");
  if (file == NULL)
  {
    printf("Matrix file not found.\n");
    exit(1);
  }

  for (i = 0; i < VOCAB_SIZE; i++)
  {
    for (j = 0; j < EMB_SIZE; j++)
    {
      fscanf(file, "%f", &(embedding_matrix[i][j]));
    }
  }
  fclose(file);

  // Create query list
  file = fopen(QUERY_FILE, "r");
  if (file == NULL)
  {
    printf("Query file not found.\n");
    exit(1);
  }

  for (i = 0; i < NUM_QUERIES; i++)
  {
    fscanf(file, "%d", &(query_list[i]));
  }
  fclose(file);


  // Partition queues
  // holds word_id's to lookup
  struct queue_root *pqueues[num_queues];
  for (j=0; j < num_queues; j++)
  {
    pqueues[j] = ALLOC_QUEUE_ROOT();
  }

  struct timespec ts;
  ts.tv_sec = 1;
  ts.tv_nsec = 10;

  struct worker_data *worker_data = malloc_aligned(sizeof(struct worker_data) * num_workers);

  // Create processes for each worker
  int iteration;
  float wps_avg;
  for (iteration=0; iteration < ITERATIONS + WARMUP; iteration++)
  {
    if (iteration < WARMUP)
    {
      printf("[W] ");
    }
    printf("Prefilling queues with queries...\n");
    for (i=0; i < prefill; i++)
    {
      int word_id = query_list[i];
      struct queue_head *item = malloc_aligned(sizeof(struct queue_head));
      INIT_QUEUE_HEAD(item, word_id, i);
      int queue_id = assign_queue(word_id, num_cold_queues, NUM_HOT_WORDS);
      //printf("Putting word_id=%i in queue=%i\n", item->word_id, queue_id);
      queue_put(item, pqueues[queue_id]);
    }
    // printf("Queues filled.\n");
  for (i=0; i < num_workers; i++)
  {
    modify_worker_count(1);

    // assign worker to a partition's queue and to the hot queue if it exists
    worker_data[i].worker_id = i;
    if (HOTCOLD_MODE == HOTCOLD_AWARE)
    {
      worker_data[i].work_queue = pqueues[i%num_cold_queues + 1];
      worker_data[i].hot_queue = pqueues[0];
      // printf("w=%d -> q=%d\n", worker_data[i].worker_id, i%num_cold_queues + 1);
    }
    else
    {
      worker_data[i].work_queue = pqueues[i % num_queues];
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
  double avg, dev, rounds_avg, rounds_dev, words_avg, words_dev;
  while (running_workers > 0)
  {
    nanosleep(&ts, NULL);
    // Single round shorter than 1 ms?
    // if (threshold / worker_data[0].words_ps < 0.001)
    // {
    //   fprintf(stderr, "threshold %lli -> %lli\n",
    //           threshold,
    //           threshold*2);
    //   threshold *= 2;
    //   continue;
    // }

    struct stddev sd, rounds, words;
    memset(&sd, 0, sizeof(sd));
    memset(&rounds, 0, sizeof(rounds));
    memset(&words, 0, sizeof(words));
    for (i=0; i < num_workers; i++)
    {
      stddev_add(&rounds, worker_data[i].rounds);
      stddev_add(&words, worker_data[i].words_ps);
    }
    // double avg, dev, rounds_avg, rounds_dev, words_avg, words_dev;
    stddev_get(&sd, NULL, &avg, &dev);
    stddev_get(&rounds, NULL, &rounds_avg, &rounds_dev);
    stddev_get(&words, NULL, &words_avg, &words_dev);
    // printf("%.3f, %.3f, %.3f, %.3f, %.3f, %llu\n", sd.sum, avg, dev, rounds_avg, rounds_dev, threshold);
    printf("wps: avg=%.3f, std=%.3f\n", words_avg, words_dev);
  }
  if (iteration >= WARMUP) { wps_avg += words_avg; }
}
wps_avg /= ITERATIONS;


  Write out response matrix
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

  printf("avg wps: %f\n", wps_avg);
  printf("%llu items processed, %d items assigned\n", total_items_processed, prefill*(ITERATIONS+WARMUP));

  return 0;
}
