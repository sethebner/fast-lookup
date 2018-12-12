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

#define WORK_STEALING_MODE 1
#define WORK_STEALING_ALLOWED 1

#define WORKERS_PER_PARTITION 1
#define THREADS_PER_WORKER 1

#define MAX_N 1000
#define MIN_N 0

#define DEBUG_MODE 1
#define DEBUG_ON 0

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

// struct thread_data
// {
//   pthread_t thread_id;
//   struct queue_root *queue;
//   double locks_ps;
//   unsigned long long rounds;
// };

struct partition_data
{
  int partition_id;
  // pthread_t worker_ids[WORKERS_PER_PARTITION];
  struct queue_root *pqueue;
  double locks_ps;
  unsigned long long rounds;
};

// struct queue_data
// {
//   int queue_id;
//   pthread_t worker_ids[WORKERS_PER_PARTITION];
//   struct queue_root *qqueue;
//   double locks_ps;
//   unsigned long long rounds;
// };

struct worker_data
{
  pthread_t thread_id;
  int worker_id;
  // pthread_t thread_ids[THREADS_PER_WORKER];
  struct queue_root *work_queue;
  struct queue_root *hot_queue;
  double locks_ps;
  double words_ps;
  unsigned long long rounds;
};

static unsigned long long threshold = 10000;

static void *worker_loop(void *_worker_data)
{
  unsigned long long counter = 0;
  unsigned long long t0, t1;
  struct worker_data *td = (struct worker_data *)_worker_data;
  struct queue_root *work_queue = td->work_queue;
  struct queue_root *hot_queue = td->hot_queue;

  struct queue_head *item = malloc_aligned(sizeof(struct queue_head));
  INIT_QUEUE_HEAD(item, -1); // dummy initialize

  t0 = gettime();
  while (1)
  {
    // prioritize "cold" words over "hot" words because cold words can be accessed only by a unique worker
    item = queue_get(work_queue);
    if (item == NULL)
    {
      if (WORK_STEALING_MODE == WORK_STEALING_ALLOWED)
      {
        // printf("Worker %i out of cold work. Reading from hot queue\n", td->worker_id);

        // pull word_id off of hot queue
        item = queue_get(hot_queue);
        if (item == NULL)
        {
          // nothing left to do
          // printf("Hot queue empty: No more work for worker %i\n", td->worker_id);

          modify_worker_count(-1);

          return NULL;
        }
        else
        {
          // printf("Worker %i processed word_id=%i from the HOTQUEUE\n", td->worker_id, item->word_id);
        }
      }
      else
      {
        // nothing left to do
        // printf("No more work for worker %i\n", td->worker_id);

        modify_worker_count(-1);

        return NULL;
      }
    }
    else
    {
      // printf("Worker %i processed word_id=%i\n", td->worker_id, item->word_id);
    }

    // TODO: look up word_id in matrix, and store in return matrix
    //    (need row index? if so, store in queue items?)
    // printf("Worker %i processed word_id=%i\n", td->worker_id, item->word_id);

    counter++;
    if (counter != threshold)
    {
      continue;
    }
    else
    {
      // profiling stats
      t1 = gettime();
      unsigned long long delta = t1 - t0;
      double locks_ps = (double)counter / ((double) delta / 1000000000LL);
      td->locks_ps = locks_ps;
      td->words_ps = (double)counter / ((double) delta / 1000000000LL);
      // printf("wps=%.3f\n", td->words_ps);
      td->rounds += 1;
      t0 = t1;
      counter = 0;

      #if 0
      printf("thread=%16lx round done in %.3fms, locks per sec=%.3f\n",
             (long)pthread_self(),
             (double)delta / 1000000,
             locks_ps);
      #endif
    }
  }

  pthread_mutex_lock(&worker_counter);
  running_workers--;
  // printf("Workers still running=%i\n", running_workers);
  pthread_mutex_unlock(&worker_counter);

  return NULL;
}

int assign_queue(int word_id, int num_partitions, int num_hot_words)
{
  if (WORK_STEALING_MODE == WORK_STEALING_ALLOWED)
  {
    if (word_id < num_hot_words) { return 0; } // hot queue available to all nodes
    else { return word_id % num_partitions + 1; } // do not assign to hot queue; assign to appropriate cold queue
  }
  else { return word_id % num_partitions; } // no hot queue, so assign to a cold queue; ignores #items in queues/load balancing
}

int get_random()
{
  return rand() % (MAX_N + 1 - MIN_N) + MIN_N;
}

int main(int argc, char **argv)
{
  int i, j;
  int num_partitions = 1; // number of partitions
  int prefill = 0; // size of batch
  int num_queues, num_workers;
  int num_hot_words = 100;

  if (argc > 1)
  {
    num_partitions = atoi(argv[1]);
  }
  if (argc > 2)
  {
    prefill = atoi(argv[2]);
  }
  if (argc > 3)
  {
    fprintf(stderr, "Usage: %s [npartitions] [prefill]\n", argv[0]);
    abort();
  }
  fprintf(stderr, "Running with npartitions=%i, prefill=%i\n", num_partitions, prefill);
  threshold /= num_partitions;

  if (WORK_STEALING_MODE == WORK_STEALING_ALLOWED)
  {
    num_queues = num_partitions + 1; // queue 0 is a privileged queue for "hot" words that all nodes may access
  }
  else
  {
    num_queues = num_partitions;
  }

  num_workers = num_partitions * WORKERS_PER_PARTITION;

  if (WORK_STEALING_MODE == WORK_STEALING_ALLOWED)
  {
    printf("Work stealing allowed. Privileged hot queue created.\n");
  }
  else
  {
    printf("Work stealing NOT allowed. Privileged hot queue NOT created.\n");
  }


  // Master queue
  // holds word_id's to assign to partition queues + hot queue if it exists
  struct queue_root *mqueue = ALLOC_QUEUE_ROOT();

  // Partition queues
  // holds word_id's to lookup
  struct queue_root *pqueues[num_queues];
  for (j=0; j < num_queues; j++)
  {
    pqueues[j] = ALLOC_QUEUE_ROOT();
  }

  for (i=0; i < prefill; i++)
  {
    int word_id = get_random(); // TODO: read word_id from file
    struct queue_head *item = malloc_aligned(sizeof(struct queue_head));
    INIT_QUEUE_HEAD(item, word_id);
    int queue_id = assign_queue(word_id, num_partitions, num_hot_words);
    //printf("Putting word_id=%i in queue=%i\n", item->word_id, queue_id);
    queue_put(item, pqueues[queue_id]);
  }

  struct partition_data *partition_data[num_queues];
  for (j=0; j < num_queues; j++)
  {
    partition_data[j] = malloc_aligned(sizeof(struct partition_data));
    partition_data[j]->partition_id = j;
    partition_data[j]->pqueue = pqueues[j];

    // TODO: assign values to rest of partition_data fields?
  }


  // Dequeue and print word_id's of queues' contents: debugging only since it removes items from queue
  if (DEBUG_MODE == DEBUG_ON)
  {
    int c;
    for (j=0; j < num_queues; j++)
    {
      printf("Queue %i\n", j);
      // struct queue_root *pqueue = pqueues[j];
      struct queue_root *pqueue = partition_data[j]->pqueue;
      struct queue_head *item = malloc_aligned(sizeof(struct queue_head));
      INIT_QUEUE_HEAD(item, -1);
      c = 0;
      while (1)
      {
        item = queue_get(pqueue);
        if (item == NULL)
        {
          // no more items in queue
          printf("%i items in queue %i\n", c, j);
          break;
        }
        c += 1;
        printf("word_id=%i\n", item->word_id);
      }
      printf("\n");
    }
  }


  struct worker_data *worker_data = malloc_aligned(sizeof(struct worker_data) * num_workers);

  // Create processes for each worker
  for (i=0; i < num_workers; i++)
  {
    modify_worker_count(1);

    // assign worker to a partition's queue and to the hot queue if it exists
    if (WORK_STEALING_MODE == WORK_STEALING_ALLOWED)
    {
      worker_data[i].work_queue = pqueues[i%num_queues + 1];
      worker_data[i].hot_queue = pqueues[0];
    }
    else
    {
      worker_data[i].work_queue = pqueues[i % num_queues];
      worker_data[i].hot_queue = NULL;
    }

    worker_data[i].worker_id = i;
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

  printf("total loops per second, average loops per thread\n");
  int reports = 0;
  while (running_workers > 0)
  {
    sleep(1);
    // Not doing any locking - we're only reading values.
    // Single round shorter than 50 ms?
    if (threshold / worker_data[0].locks_ps < 0.05)
    {
      fprintf(stderr, "threshold %lli -> %lli\n",
              threshold,
              threshold*2);
      threshold *= 2;
      continue;
    }

    struct stddev sd, rounds, words;
    memset(&sd, 0, sizeof(sd));
    memset(&rounds, 0, sizeof(rounds));
    memset(&words, 0, sizeof(words));
    for (i=0; i < num_workers; i++)
    {
      stddev_add(&sd, worker_data[i].locks_ps);
      stddev_add(&rounds, worker_data[i].rounds);
      stddev_add(&words, worker_data[i].words_ps);
    }
    double avg, dev, rounds_avg, rounds_dev, words_avg, words_dev;
    stddev_get(&sd, NULL, &avg, &dev);
    stddev_get(&rounds, NULL, &rounds_avg, &rounds_dev);
    stddev_get(&words, NULL, &words_avg, &words_dev);
    // printf("%.3f, %.3f, %.3f, %.3f, %.3f, %llu\n", sd.sum, avg, dev, rounds_avg, rounds_dev, threshold);
    printf("wps: avg=%.3f, std=%.3f\n", words_avg, words_dev);
    if (reports++ > 20)
    {
      break;
    }
  }

  return 0;
}
