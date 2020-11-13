#include "page_rank.h"

#include <stdlib.h>
#include <cmath>
#include <omp.h>
#include <utility>

#include "../common/CycleTimer.h"
#include "../common/graph.h"

#include <vector>

// pageRank --
//
// g:           graph to process (see common/graph.h)
// solution:    array of per-vertex vertex scores (length of array is num_nodes(g))
// damping:     page-rank algorithm's damping parameter
// convergence: page-rank algorithm's convergence threshold
//
void pageRank(Graph g, double* solution, double damping, double convergence)
{

  // initialize vertex weights to uniform probability. Double
  // precision scores are used to avoid underflow for large graphs

  // int numNodes = num_nodes(g);
  // double equal_prob = 1.0 / numNodes;
  // for (int i = 0; i < numNodes; ++i) {
  //   solution[i] = equal_prob;
  // }
  int numNodes = num_nodes(g);
  double equal_prob = 1.0 / numNodes;
#pragma omp parallel for
  for (int i = 0; i < numNodes; ++i) {
    solution[i] = equal_prob;
  }

  /* 418/618 Students: Implement the page rank algorithm here.  You
     are expected to parallelize the algorithm using openMP.  Your
     solution may need to allocate (and free) temporary arrays.

     Basic page rank pseudocode:

     // initialization: see example code above
     score_old[vi] = 1/numNodes;

     while (!converged) {

       // compute score_new[vi] for all nodes vi:
       score_new[vi] = sum over all nodes vj reachable from incoming edges
                          { score_old[vj] / number of edges leaving vj  }
       score_new[vi] = (damping * score_new[vi]) + (1.0-damping) / numNodes;

       score_new[vi] += sum over all nodes vj with no outgoing edges
                          { damping * score_old[vj] / numNodes }

       // compute how much per-node scores have changed
       // quit once algorithm has converged

       global_diff = sum over all nodes vi { abs(score_new[vi] - score_old[vi]) };
       converged = (global_diff < convergence)
     }

   */
  // // vector<int> nodes_with_no_outgoing_edge;
  // // for (int vi = 0; vi < numNodes; ++vi) {
  // //   if (outgoing_size(g, vi) <= 0)
  // //     nodes_with_no_outgoing_edge.push_back(vi);
  // // }
  // bool converged = false;
  // double damping_value = (1.0 - damping) / numNodes;
  // while (!converged) {
  //   double global_diff = 0;
  //   for (int vi = 0; vi < numNodes; ++vi) {
  //     const Vertex* start = incoming_begin(g, vi);
  //     const Vertex* end = incoming_end(g, vi);
  //     double score_new = 0;
  //     for (const Vertex* vj = start; vj != end; ++vj) {
  //       score_new += solution[*vj] / outgoing_size(g, *vj);
  //     }
  //     score_new = (damping * score_new) + damping_value;
  //     // for (int i = 0; i < nodes_with_no_outgoing_edge.size(); ++i) {
  //     //   score_new += damping * solution[nodes_with_no_outgoing_edge[i]] / numNodes;
  //     // }
  //     global_diff += abs(score_new - solution[vi]);
  //     solution[vi] = score_new;
  //   }
  //   converged = global_diff < convergence;
  // }


//   int thread_count = 1;
//   bool converged = false;
//   double damping_value = (1.0 - damping) / numNodes;
//   ///iteration untill global diff less than convergence
//   while (!converged) {
//     double global_diff = 0;
// #pragma omp parallel for num_threads(thread_count)
// // #pragma omp parallel for num_threads(thread_count) schedule(static, 1)
//     for (int vi = 0; vi < numNodes; ++vi) {
//       ///loop over all the incoming edge 's node
//       const Vertex* start = incoming_begin(g, vi);
//       const Vertex* end = incoming_end(g, vi);
//       double score_new = 0;
//       for (const Vertex* vj = start; vj != end; ++vj) {
//         score_new += solution[*vj] / outgoing_size(g, *vj);
//       }
//       score_new = (damping * score_new) + damping_value;
// #pragma omp critical
//       global_diff += abs(score_new - solution[vi]);
// // #pragma omp atomic ///does it matter if reading an old value?
//       solution[vi] = score_new;
//     }
//     converged = global_diff < convergence;
//   }

//   std::vector<int> nodes_with_no_outgoing_edge;
//   ///stl container is not thread-safe; concurrent write can't be parallelized
//   for (int vi = 0; vi < numNodes; ++vi) {
//     if (outgoing_size(g, vi) <= 0)
//       nodes_with_no_outgoing_edge.push_back(vi);
//   }
//   bool converged = false;
//   double damping_value = (1.0 - damping) / numNodes;
//   double* score_new = new double[numNodes];
//   while (!converged) {
//     double global_diff = 0;
// #pragma omp parallel for
// // #pragma omp parallel for num_threads(thread_count) schedule(static, 1)
//     for (int vi = 0; vi < numNodes; ++vi) {
//       ///loop over all the incoming edge 's node
//       const Vertex* start = incoming_begin(g, vi);
//       const Vertex* end = incoming_end(g, vi);
//       for (const Vertex* vj = start; vj != end; ++vj) {
//         score_new[vi] += solution[*vj] / outgoing_size(g, *vj);
//       }
//       score_new[vi] = (damping * score_new[vi]) + damping_value;
//       ///loop over all the nodes with no outgoing edge
//       for (int i = 0; i < nodes_with_no_outgoing_edge.size(); ++i) {
//         score_new[vi] += damping * solution[nodes_with_no_outgoing_edge[i]] / numNodes;
//       }
// #pragma omp critical
//       global_diff += abs(score_new[vi] - solution[vi]);
//     }
// #pragma omp parallel for
//   for (int vi = 0; vi < numNodes; ++vi) {
//     solution[vi] = score_new[vi];///update solution
//     score_new[vi] = 0;
//   }
//     converged = global_diff < convergence;
//   }

//   std::vector<int> nodes_with_no_outgoing_edge;
//   for (int vi = 0; vi < numNodes; ++vi) {
//     if (outgoing_size(g, vi) <= 0)
//       nodes_with_no_outgoing_edge.push_back(vi);
//   }
//   bool converged = false;
//   double damping_value = (1.0 - damping) / numNodes;

//   while (!converged) {
//     double global_diff = 0;
// #pragma omp parallel for reduction(+ : global_diff)
// // #pragma omp parallel for num_threads(thread_count) schedule(static, 1)
//     for (int vi = 0; vi < numNodes; ++vi) {
//       ///loop over all the incoming edge 's node
//       const Vertex* start = incoming_begin(g, vi);
//       const Vertex* end = incoming_end(g, vi);
//       double score_new = 0;
//       for (const Vertex* vj = start; vj != end; ++vj) {
//         score_new += solution[*vj] / outgoing_size(g, *vj);
//       }
//       score_new = (damping * score_new) + damping_value;
//       ///loop over all the nodes with no outgoing edge
//       for (int i = 0; i < nodes_with_no_outgoing_edge.size(); ++i) {
//         score_new += damping * solution[nodes_with_no_outgoing_edge[i]] / numNodes;
//       }
//       global_diff += abs(score_new - solution[vi]);
//       solution[vi] = score_new;
//     }
//     converged = global_diff < convergence;
//   }

  std::vector<int> nodes_with_no_outgoing_edge;
  ///stl container is not thread-safe; concurrent write can't be parallelized
  for (int vi = 0; vi < numNodes; ++vi) {
    if (outgoing_size(g, vi) <= 0)
      nodes_with_no_outgoing_edge.push_back(vi);
  }
  bool converged = false;
  double damping_value = (1.0 - damping) / numNodes;
  double* score_new = new double[numNodes];
  while (!converged) {
    double global_diff = 0;
#pragma omp parallel for reduction(+ : global_diff)
// #pragma omp parallel for num_threads(thread_count) schedule(static, 1)
    for (int vi = 0; vi < numNodes; ++vi) {
      ///loop over all the incoming edge 's node
      const Vertex* start = incoming_begin(g, vi);
      const Vertex* end = incoming_end(g, vi);
      for (const Vertex* vj = start; vj != end; ++vj) {
        score_new[vi] += solution[*vj] / outgoing_size(g, *vj);
      }
      score_new[vi] = (damping * score_new[vi]) + damping_value;
      ///loop over all the nodes with no outgoing edge
      for (int i = 0; i < nodes_with_no_outgoing_edge.size(); ++i) {
        score_new[vi] += damping * solution[nodes_with_no_outgoing_edge[i]] / numNodes;
      }
      global_diff += abs(score_new[vi] - solution[vi]);
    }
#pragma omp parallel for
  for (int vi = 0; vi < numNodes; ++vi) {
    solution[vi] = score_new[vi];///update solution
    score_new[vi] = 0;
  }
    converged = global_diff < convergence;
  }
}
