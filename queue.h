// https://github.com/majek/dump/blob/master/msqueue/queue.h

#ifndef QUEUE_H
#define QUEUE_H

struct queue_root;

struct queue_head
{
  struct queue_head *next;
  int word_id;
  int row_id;
};

struct queue_root *ALLOC_QUEUE_ROOT();
void INIT_QUEUE_HEAD(struct queue_head *head, int word_id, int row_id);

void queue_put(struct queue_head *new,
               struct queue_root *root);

struct queue_head *queue_get(struct queue_root *root);
int queue_get_n(struct queue_root *root, struct queue_head *workload[], int n, int worker_id);

#endif // QUEUE_H
