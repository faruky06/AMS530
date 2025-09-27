#include <mpi.h>
#include<iostream>

int main(int argc, char *argv[]) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    MPI_Request req_send, req_recv;
    int token;
    //receive(except for rank 0)
    if (rank != 0)
        MPI_Irecv(&token, 1, MPI_INT, rank-1, 0, MPI_COMM_WORLD, &req_recv);
    //print message immediately for rank 0, or after receiving token for others
    if (rank == 0) {
        std::cout << "Hello from " << rank << "of " << size << std::endl;
    } else {
        MPI_Wait(&req_recv, MPI_STATUS_IGNORE); //wait until token arrives
        std::cout << "Hello from " << rank << "of " << size << std::endl;
    }
    //send
    if (rank < size - 1)
        MPI_Isend(&rank, 1, MPI_INT, rank+1, 0, MPI_COMM_WORLD, &req_send);
    //for send to complete before finishing
    if (rank < size - 1)
        MPI_Wait(&req_send, MPI_STATUS_IGNORE);
    MPI_Finalize();
}