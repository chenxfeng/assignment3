#include <cstring>
#include <set>
#include <iostream>
#include <vector>
#include <queue>
#include "bfs.h"

#define contains(container, element) \
  (container.find(element) != container.end())

/**
 *
 * global_frontier_sync--
 * 
 * Takes a distributed graph, and a distributed frontier with each node containing
 * world_size independently produced new frontiers, and merges them such that each
 * node holds the subset of the global frontier containing local vertices.
 */
void global_frontier_sync(DistGraph &g, DistFrontier &frontier, int *depths) {

  // TODO 15-418/618 STUDENTS
  //
  // In this function, you should synchronize between all nodes you
  // are using for your computation. This would mean sending and
  // receiving data between nodes in a manner you see fit. Note for
  // those using async sends: you should be careful to make sure that
  // any data you send is received before you delete or modify the
  // buffers you are sending.

  int world_size = g.world_size;
  int world_rank = g.world_rank;

  std::vector<int*> send_bufs;
  std::vector<int> send_idx;
  std::vector<int*> recv_bufs;
  MPI_Request* send_reqs = new MPI_Request[world_size];
  MPI_Status* probe_status = new MPI_Status[world_size];
  ///broadcast part next_frontier to every other processes
  for (int i = 0; i < world_size; i++) {
    if (i != world_rank) {
      int msglen = frontier.sizes[i] == 0 ? 1 : frontier.sizes[i]*2;
      int* send_buf = new int[msglen];
      send_bufs.push_back(send_buf);
      send_idx.push_back(i);
      ///fill if no empty
      if (msglen != 1) {
        for (int j = 0; j < frontier.sizes[i]; ++j) {
          send_buf[2*j] = frontier.elements[i][j];
          send_buf[2*j+1] = frontier.depths[i][j];
        }
      }
      MPI_Isend(send_buf, msglen, MPI_INT,
                i, 0, MPI_COMM_WORLD, &send_reqs[i]);
    }
  }
  ///recv local part of next_frontier from all other processes
  for (int i = 0; i < world_size; i++) {
    if (i != world_rank) {
      ///probe and wait for message from i
      MPI_Status status;
      MPI_Probe(i, 0, MPI_COMM_WORLD, &probe_status[i]);
      int num_vals = 0;
      MPI_Get_count(&probe_status[i], MPI_INT, &num_vals);
      ///prepare recv buffer
      int* recv_buf = new int[num_vals];
      recv_bufs.push_back(recv_buf);
      MPI_Recv(recv_buf, num_vals, MPI_INT, probe_status[i].MPI_SOURCE,
               probe_status[i].MPI_TAG, MPI_COMM_WORLD, &status);
      if (num_vals != 1) {
        for (int j = 0; j < num_vals/2; ++j) {
          ///check whether visited
          int node = recv_buf[2*j];
          if (depths[node - g.start_vertex] == NOT_VISITED_MARKER) {
            frontier.add(world_rank, node, recv_buf[2*j+1]);
            depths[node - g.start_vertex] = recv_buf[2*j+1];
          }
        }
      }
    }
  }
  ///check whether messages sent are all received
  for (size_t i = 0; i < send_bufs.size(); i++) {
    MPI_Status status;
    MPI_Wait(&send_reqs[send_idx[i]], &status);
    delete(send_bufs[i]);
  }
  for (size_t i = 0; i < recv_bufs.size(); i++) {
    delete(recv_bufs[i]);
  }
  delete(send_reqs);
  delete(probe_status);
}

/*
 * bfs_step --
 * 
 * Carry out one step of a distributed bfs
 * 
 * depths: current state of depths array for local vertices
 * current_frontier/next_frontier: copies of the distributed frontier structure
 * 
 * NOTE TO STUDENTS: We gave you this function as a stub.  Feel free
 * to change as you please (including the arguments)
 */
void bfs_step(DistGraph &g, int *depths,
	      DistFrontier &current_frontier,
              DistFrontier &next_frontier) {

  int frontier_size = current_frontier.get_local_frontier_size();
  Vertex* local_frontier = current_frontier.get_local_frontier();

  // keep in mind, this node owns the vertices with global ids:
  // g.start_vertex, g.start_vertex+1, g.start_vertex+2, etc...

  // TODO 15-418/618 STUDENTS
  //
  // implement a step of the BFS
  for (int i = 0; i < frontier_size; ++i) {
    Vertex node = local_frontier[i];
    // printf("vertex %d depth %d in local_frontier\n", node, depths[node - g.start_vertex]);
    ///loop ver node's all outgoing neighbor
    for (size_t neighbor = 0; neighbor < g.v_out_edges[node - g.start_vertex].size(); ++neighbor) {
      int outgoing = g.v_out_edges[node - g.start_vertex][neighbor];
      ///assume vertex not visited: add to next_frontier, check when sync
      if (g.get_vertex_owner_rank(outgoing) != g.world_rank)
        next_frontier.add(g.get_vertex_owner_rank(outgoing), outgoing, depths[node - g.start_vertex]+1);
      if (g.get_vertex_owner_rank(outgoing) == g.world_rank 
        && depths[outgoing - g.start_vertex] == NOT_VISITED_MARKER) {
        next_frontier.add(g.world_rank, outgoing, depths[node - g.start_vertex]+1);
        depths[outgoing - g.start_vertex] = depths[node - g.start_vertex] + 1;
      }
    }
  }
}

/*
 * bfs --
 * 
 * Execute a distributed BFS on the distributed graph g
 * 
 * Upon return, depths[i] should be the distance of the i'th local
 * vertex from the BFS root node
 */
void bfs(DistGraph &g, int *depths) {
  DistFrontier current_frontier(g.vertices_per_process, g.world_size,
                                g.world_rank);
  DistFrontier next_frontier(g.vertices_per_process, g.world_size,
                             g.world_rank);

  DistFrontier *cur_front = &current_frontier,
               *next_front = &next_frontier;

  // Initialize all the depths to NOT_VISITED_MARKER.
  // Note: Only storing local vertex depths.
  for (int i = 0; i < g.vertices_per_process; ++i )
    depths[i] = NOT_VISITED_MARKER;

  // Add the root node to the frontier 
  int offset = g.start_vertex;
  if (g.get_vertex_owner_rank(ROOT_NODE_ID) == g.world_rank) {
    current_frontier.add(g.get_vertex_owner_rank(ROOT_NODE_ID), ROOT_NODE_ID, 0);
    depths[ROOT_NODE_ID - offset] = 0;
  }

  while (true) {
    // printf("iteration begin from process %d\n", g.world_rank);

    bfs_step(g, depths, *cur_front, *next_front);

    // this is a global empty check, not a local frontier empty check.
    // You will need to implement is_empty() in ../dist_graph.h
    // if (next_front->is_empty())
    //   break;
    int cover_local = 1;
    for (int i = 0; i < g.vertices_per_process; ++i) {
      if (depths[i] == NOT_VISITED_MARKER && ///take out vertex without incoming edge
        g.v_no_in_edge.count(i+g.vertices_per_process*g.world_rank) == 0) {
        cover_local = 0;
        break;
      }
    }
    // for (int i = 0; i < g.vertices_per_process; ++i) {
    //   printf("vertex %d: depth %d\n", i+g.vertices_per_process*g.world_rank, depths[i]);
    // }
    // printf("iteration m1 from process %d: local %d\n", g.world_rank, cover_local);

    int cover_all = 0;
    // MPI_Allreduce(&cover_local, &cover_all, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&cover_local, &cover_all, 1, MPI_INT, MPI_LAND, MPI_COMM_WORLD);
    // printf("iteration m2 from process %d: gobal %d\n", g.world_rank, cover_all);
    if (cover_all == 1)//g.world_size)
      break;
    // exchange frontier information
    global_frontier_sync(g, *next_front, depths);

    // for (int i = 0; i < g.vertices_per_process; ++i) {
    //   printf("after sync vertex %d: depth %d\n", i+g.vertices_per_process*g.world_rank, depths[i]);
    // }

    DistFrontier *temp = cur_front;
    cur_front = next_front;
    next_front = temp;
    next_front -> clear();

    // printf("iteration end from process %d\n", g.world_rank);
  }
  if (g.world_rank == 0) printf("vfs finish %d\n", g.world_rank);
}

