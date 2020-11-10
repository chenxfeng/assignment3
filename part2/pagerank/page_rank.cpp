#include "page_rank.h"

/*
 * pageRank-- 
 *
 * Computes page rank on a distributed graph g
 * 
 * Per-vertex scores for all vertices *owned by this node* (not all
 * vertices in the graph) should be placed in `solution` upon
 * completion.
 */
void pageRank(DistGraph &g, double* solution, double damping, double convergence) {

    // TODO FOR 15-418/618 STUDENTS:

    // Implement the distributed page rank algorithm here. This is
    // very similar to what you implemnted in Part 1, except in this
    // case, the graph g is distributed across cluster nodes.

    // Each node in the cluster is only aware of the outgoing edge
    // topology for the vertices it "owns".  The cluster nodes will
    // need to coordinate to determine what information.

    // note: we give you starter code below to initialize scores for
    // ALL VERTICES in the graph, but feel free to modify as desired.
    // Keep in mind the `solution` array returned to the caller should
    // only have scores for the local vertices
    int totalVertices = g.total_vertices();
    double equal_prob = 1.0/totalVertices;

    int vertices_per_process = g.vertices_per_process;

    std::vector<double> score_curr(totalVertices);
    std::vector<double> score_next(g.vertices_per_process);

    // initialize per-vertex scores
    #pragma omp parallel for
    for (Vertex i = 0; i < totalVertices; i++) {
        score_curr[i] = equal_prob;
    }

    bool converged = false;
    double damping_value = (1.0 - damping) / totalVertices;

    /*

      Repeating basic pagerank pseudocode here for your convenience
      (same as for part 1 of this assignment)

    while (!converged) {

        // compute score_new[vi] for all vertices belonging to this process
        score_new[vi] = sum over all vertices vj reachable from incoming edges
                          { score_old[vj] / number of edges leaving vj  }
        score_new[vi] = (damping * score_new[vi]) + (1.0-damping) / totalVertices;

        score_new[vi] += sum over all nodes vj with no outgoing edges
                          { damping * score_old[vj] / totalVertices }

        // compute how much per-node scores have changed
        // quit once algorithm has converged

        global_diff = sum over all vertices vi { abs(score_new[vi] - score_old[vi]) };
        converged = (global_diff < convergence)

        // Note that here, some communication between all the nodes is necessary
        // so that all nodes have the same copy of old scores before beginning the 
        // next iteration. You should be careful to make sure that any data you send 
        // is received before you delete or modify the buffers you are sending.

    }

    // Fill in solution with the scores of the vertices belonging to this node.

    */
    while (!converged) {
        double local_diff = 0;///need mpi_all_reduce
        for (int vi = g.start_vertex; vi <= g.end_vertex; ++vi) {
            score_next[vi - g.start_vertex] = 0;
            ///loop over all the incoming edge 's node of local vertex
            for (int i = 0; i < g.v_in_edges[vi - g.start_vertex].size(); ++i) {
                score_next[vi - g.start_vertex] += score_curr[g.v_in_edges[vi - g.start_vertex][i]] 
                                            / g.v_to_out_degree[g.v_in_edges[vi - g.start_vertex][i]];
            }
            score_next[vi - g.start_vertex] = damping * score_next[vi - g.start_vertex] + damping_value;
            ///loop over all the nodes with no outgoing edge
            for (int i = 0; i < g.v_no_out_edge.size(); ++i) {
                score_next[vi - g.start_vertex] += damping * score_curr[g.v_no_out_edge[i]] / totalVertices;
            }
            local_diff += fabs(score_next[vi - g.start_vertex] - score_curr[vi]);
            score_curr[vi] = score_next[vi - g.start_vertex];
        }
        ///all reduce the local_diff value to global_diff
        double global_diff;
        MPI_Allreduce(&local_diff, &global_diff, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        converged = global_diff < convergence;
        ///communicate for result of this iteration
        if (!converged) {
if (g.world_rank == 0) printf("communication begin\n");
            double * send_buf = score_next.data();
            double * recv_bufs = score_curr.data();            
            ///bcast new score of local vertex
            MPI_Request* send_reqs = new MPI_Request[g.world_size];
            for (int i = 0; i < g.world_size; ++i) {
                if (g.send_process_ids.count(i)) {
                    if (g.world_rank == 0) printf("send %d\n", i);
                    MPI_Isend(send_buf, vertices_per_process, MPI_DOUBLE, 
                        i, 0, MPI_COMM_WORLD, &send_reqs[i]);
                }
            }
if (g.world_rank == 0) printf("iteration\n");
            ///recv new score from other nodes
            MPI_Status* probe_status = new MPI_Status[g.world_size];
            for (int i = 0; i < g.world_size; ++i) {
                if (g.recv_process_ids.count(i)) {
                    ///probe and wait for message from process i
                    MPI_Status status;
                    MPI_Probe(i, 0, MPI_COMM_WORLD, &probe_status[i]);
                    int num_vals = 0;///must be equal to vertices_per_process
                    MPI_Get_count(&probe_status[i], MPI_DOUBLE, &num_vals);
                    assert(num_vals == vertices_per_process);
                    MPI_Recv(recv_bufs + vertices_per_process*i, num_vals, MPI_DOUBLE,
                        probe_status[i].MPI_SOURCE, probe_status[i].MPI_TAG, MPI_COMM_WORLD, &status);
                }
            }
            ///check whether messages sent are all received
            for (int i = 0; i < g.world_size; ++i) {
                if (g.send_process_ids.count(i)) {
                    MPI_Status status;
                    MPI_Wait(&send_reqs[i], &status);
                }
            }
            delete(send_reqs);
            delete(probe_status);
        }

        if (g.world_rank == 0) 
            printf("global_diff: %f, local_diff: %f, %f\n", global_diff, local_diff, convergence);
    }
//     while (!converged) {
//         double local_diff = 0;///need mpi_all_reduce
// #pragma omp parallel for
// // #pragma omp parallel for num_threads(thread_count) schedule(static, 1)
//         for (int vi = g.start_vertex; vi <= g.end_vertex; ++vi) {
//             score_next[vi - g.start_vertex] = 0;
//             ///loop over all the incoming edge 's node of local vertex
//             for (int i = 0; i < g.v_in_edges[vi - g.start_vertex].size(); ++i) {
//                 score_next[vi - g.start_vertex] += score_curr[g.v_in_edges[vi - g.start_vertex][i]] 
//                                             / g.v_to_out_degree[g.v_in_edges[vi - g.start_vertex][i]];
//             }
//             score_next[vi - g.start_vertex] = damping * score_next[vi - g.start_vertex] + damping_value;
//             ///loop over all the nodes with no outgoing edge
//             for (int i = 0; i < g.v_no_out_edge.size(); ++i) {
//                 score_next[vi - g.start_vertex] += damping * score_curr[g.v_no_out_edge[i]] / totalVertices;
//             }
// #pragma omp critical
//             local_diff += abs(score_next[vi - g.start_vertex] - score_curr[vi]);
// // #pragma omp critical ///does it matter if reading an old value?
//             score_curr[vi] = score_next[vi - g.start_vertex];
//         }
//         ///all reduce the local_diff value to global_diff
//         double global_diff = 0;
//         MPI_Allreduce(&local_diff, &global_diff, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
//         converged = global_diff < convergence;
//         ///communicate for result of this iteration
//         if (!converged) {
//             double * send_buf = score_next.data();
//             double * recv_bufs = score_curr.data();            
//             ///bcast new score of local vertex
//             MPI_Request* send_reqs = new MPI_Request[g.world_size];
//             for (int i = 0; i < g.world_size; ++i) {
//                 if (g.send_process_ids.count(i)) {
//                     MPI_Isend(send_buf, vertices_per_process, MPI_DOUBLE, 
//                         i, 0, MPI_COMM_WORLD, &send_reqs[i]);
//                 }
//             }
//             ///recv new score from other nodes
//             MPI_Status* probe_status = new MPI_Status[g.world_size];
//             for (int i = 0; i < g.world_size; ++i) {
//                 if (g.recv_process_ids.count(i)) {
//                     ///probe and wait for message from process i
//                     MPI_Status status;
//                     MPI_Probe(i, 0, MPI_COMM_WORLD, &probe_status[i]);
//                     int num_vals = 0;///must be equal to vertices_per_process
//                     MPI_Get_count(&probe_status[i], MPI_DOUBLE, &num_vals);
//                     assert(num_vals == vertices_per_process);
//                     MPI_Recv(recv_bufs + vertices_per_process*i, num_vals, MPI_DOUBLE,
//                         probe_status[i].MPI_SOURCE, probe_status[i].MPI_TAG, MPI_COMM_WORLD, &status);
//                 }
//             }
//             ///check whether messages sent are all received
//             for (int i = 0; i < g.world_size; ++i) {
//                 if (g.send_process_ids.count(i)) {
//                     MPI_Status status;
//                     MPI_Wait(&send_reqs[i], &status);
//                 }
//             }
//             delete(send_reqs);
//             delete(probe_status);
//         }

//         if (g.world_rank == 0) printf("global_diff: %f, local_diff: %f\n", global_diff, local_diff);
//     }
}
