#ifndef __BFS_H__
#define __BFS_H__

//#define DEBUG

#include <set>

#include "../common/graph.h"

struct solution
{
  int *distances;
};

struct vertex_set {
  // # of vertices in the set
  int count;
  // max size of buffer vertices 
  int max_vertices;
  // array of vertex ids in set
  int *vertices;

  ///for quickly query
  std::set<int> query;
};


void bfs_top_down(Graph graph, solution* sol);
void bfs_bottom_up(Graph graph, solution* sol);
void bfs_hybrid(Graph graph, solution* sol);

#endif
