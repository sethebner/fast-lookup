// https://github.com/majek/dump/blob/master/msqueue/queue.h

#ifndef QUEUE_H
#define QUEUE_H

struct queue_root;

struct row_id_list
{
  struct row_id_list *next;
  int row_id;
};

struct queue_head
{
  struct queue_head *next;
  int word_id;
  struct row_id_list *row_head;
};

struct queue_root *ALLOC_QUEUE_ROOT();
void INIT_QUEUE_HEAD(struct queue_head *head, int word_id, struct row_id_list *positions_head);

void queue_put(struct queue_head *new,
               struct queue_root *root);

struct queue_head *queue_get(struct queue_root *root);
int queue_get_n(struct queue_root *root, struct queue_head *workload[], int n, int worker_id, int locking);

void push(struct row_id_list **head_ref, int new_data);

#endif // QUEUE_H
