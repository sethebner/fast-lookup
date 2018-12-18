// https://github.com/majek/dump/blob/master/msqueue/queue_lock_mutex.c

#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>

#include "queue.h"

#define QUEUE_POISON1 ((void*)0xDEADBEEF)

struct queue_root
{
  struct queue_head *head;
  pthread_mutex_t head_lock;

  struct queue_head *tail;
  pthread_mutex_t tail_lock;

  struct queue_head divider; // sentinel
};

struct queue_root *ALLOC_QUEUE_ROOT()
{
  struct queue_root *root = malloc(sizeof(struct queue_root));
  pthread_mutex_init(&root->head_lock, NULL);
  pthread_mutex_init(&root->tail_lock, NULL);

  root->divider.next = NULL;
  root->head = &root->divider;
  root->tail = &root->divider;
  return root;
}

void INIT_QUEUE_HEAD(struct queue_head *head, int word_id, int row_id)
{
  head->next = QUEUE_POISON1;
  head->word_id = word_id;
  head->row_id = row_id;
}

void queue_put(struct queue_head *new,
               struct queue_root *root)
{
  new->next = NULL;

  pthread_mutex_lock(&root->tail_lock);
  root->tail->next = new;
  root->tail = new;
  pthread_mutex_unlock(&root->tail_lock);
}

struct queue_head *queue_get(struct queue_root *root)
{
  struct queue_head *head, *next;

  while (1)
  {
    pthread_mutex_lock(&root->head_lock);
    head = root->head;
    next = head->next;
    if (next == NULL)
    {
      pthread_mutex_unlock(&root->head_lock);
      return NULL;
    }
    root->head = next;
    pthread_mutex_unlock(&root->head_lock);

    if (head == &root->divider)
    {
      queue_put(head, root);
      continue;
    }

    head->next = QUEUE_POISON1;
    return head;
  }
}

int queue_get_n(struct queue_root *root, struct queue_head **workload, int n, int worker_id)
{
  /*
    Return up to n items from the queue (stored in `workload`).
  */
  // struct queue_head *head, *next;
  //
  // int i;
  // int retrieved = 0;
  // while (1)
  // {
  //   pthread_mutex_lock(&root->head_lock);
  //   for (i = 0; i < n; i++)
  //   {
  //     head = root->head;
  //     next = head->next;
  //     if (next == NULL)
  //     {
  //       // No more items on queue
  //       pthread_mutex_unlock(&root->head_lock);
  //       printf("%d:queue empty\n", worker_id);
  //       return retrieved;
  //     }
  //     root->head = next;
  //     //pthread_mutex_unlock(&root->head_lock);
  //
  //     if (head == &root->divider)
  //     {
  //       queue_put(head, root);
  //       pthread_mutex_unlock(&root->head_lock);
  //       printf("%d:reached divider\n", worker_id);
  //       return retrieved;
  //     }
  //
  //     head->next = QUEUE_POISON1;
  //     workload[i] = head;
  //     printf("%d::workload[%d] gets word id %d\n", worker_id, i, head->word_id);
  //     retrieved += 1;
  //   }
  //   pthread_mutex_unlock(&root->head_lock);
  //   return retrieved;
  // }


  struct queue_head *head, *next;

  int i;
  int retrieved = 0;
  pthread_mutex_lock(&root->head_lock);
  for (i = 0; i < n; i++)
  {
    // Retrieve a queue item
    while (1)
    {
      head = root->head;
      next = head->next;
      if (next == NULL)
      {
        // No more items on queue
        pthread_mutex_unlock(&root->head_lock);
        // printf("%d:queue empty\n", worker_id);
        return retrieved;
      }
      root->head = next;
      // pthread_mutex_unlock(&root->head_lock);

      if (head == &root->divider)
      {
        queue_put(head, root);
        // pthread_mutex_unlock(&root->head_lock);
        // printf("%d:reached divider\n", worker_id);
        continue;
      }

      head->next = QUEUE_POISON1;
      workload[i] = head;
      retrieved += 1;
      // printf("%d::workload[%d] gets word id %d\n", worker_id, i, head->word_id);
      break; // populate next element of `workload`
    }
  }
  pthread_mutex_unlock(&root->head_lock);
  return retrieved;


  // struct queue_head *head, *next;
  //
  // while (1)
  // {
  //   pthread_mutex_lock(&root->head_lock);
  //   head = root->head;
  //   next = head->next;
  //   if (next == NULL)
  //   {
  //     pthread_mutex_unlock(&root->head_lock);
  //     return NULL;
  //   }
  //   root->head = next;
  //   pthread_mutex_unlock(&root->head_lock);
  //
  //   if (head == &root->divider)
  //   {
  //     queue_put(head, root);
  //     continue;
  //   }
  //
  //   head->next = QUEUE_POISON1;
  //   return head;
  // }
}
