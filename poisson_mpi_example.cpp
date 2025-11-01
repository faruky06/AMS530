//oisson_mpi_example_random.cpp
//compile: mpic++ -O3 -std=c++11 poisson_mpi_example.cpp -o poisson_mpi_example
//run (example): mpirun -np 4 ./poisson_mpi_example 128 128 5000 jacobi 100

#include <mpi.h>
#include <vector>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <string>
#include <algorithm>
#include <cstdlib> //for rand()
#include <ctime>   //for time()

inline int idx(int i, int j, int nx_with_ghost) { return i * nx_with_ghost + j; }

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, P;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &P);

    if (argc < 6) {
        if (rank == 0) std::cerr << "Usage: " << argv[0] << " nx ny max_iters method(jacobi|rbgs) print_every\n";
        MPI_Finalize();
        return 1;
    }

    int nx = std::max(4, std::atoi(argv[1]));
    int ny = std::max(4, std::atoi(argv[2]));
    int max_iters = std::atoi(argv[3]);
    std::string method = argv[4];
    int print_every = std::atoi(argv[5]);

    double dx = 1.0 / (nx + 1);
    double dy = 1.0 / (ny + 1);
    double dx2 = dx*dx;
    double dy2 = dy*dy;
    double denom = 2.0*(dx2 + dy2);

    int base = ny / P;
    int rem = ny % P;
    int local_ny = base + (rank < rem ? 1 : 0);
    int start_row = rank * base + std::min(rank, rem) + 1;

    int nx_with_ghost = nx + 2;
    int local_rows_with_ghost = local_ny + 2;

    std::vector<double> u(local_rows_with_ghost*nx_with_ghost, 0.0);
    std::vector<double> u_new(local_rows_with_ghost*nx_with_ghost, 0.0);

    
    //random initial guess
    std::srand(std::time(nullptr) + rank); //different seed per rank
    for (int i=1;i<=local_ny;++i) {
        for (int j=1;j<=nx;++j) {
            u[idx(i,j,nx_with_ghost)] = ((double)std::rand() / RAND_MAX) * 0.1; //random guess
        }
    }

    int up_rank = (rank == 0 ? MPI_PROC_NULL : rank-1);
    int down_rank = (rank == P-1 ? MPI_PROC_NULL : rank+1);

    auto exchange_halos = [&](std::vector<double>& arr) {
        MPI_Sendrecv(&arr[idx(1,0,nx_with_ghost)], nx_with_ghost, MPI_DOUBLE, up_rank, 0,
                     &arr[idx(local_rows_with_ghost-1,0,nx_with_ghost)], nx_with_ghost, MPI_DOUBLE, down_rank, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        MPI_Sendrecv(&arr[idx(local_ny,0,nx_with_ghost)], nx_with_ghost, MPI_DOUBLE, down_rank, 1,
                     &arr[idx(0,0,nx_with_ghost)], nx_with_ghost, MPI_DOUBLE, up_rank, 1,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    };

    auto compute_local_residual = [&]() {
        double local_max = 0.0;
        for (int i=1;i<=local_ny;++i) {
            for (int j=1;j<=nx;++j) {
                double lap = (u[idx(i-1,j,nx_with_ghost)] - 2.0*u[idx(i,j,nx_with_ghost)] + u[idx(i+1,j,nx_with_ghost)])/dy2
                           + (u[idx(i,j-1,nx_with_ghost)] - 2.0*u[idx(i,j,nx_with_ghost)] + u[idx(i,j+1,nx_with_ghost)])/dx2;
                local_max = std::max(local_max, std::abs(lap));
            }
        }
        double global_max;
        MPI_Allreduce(&local_max, &global_max, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        return global_max;
    };

    exchange_halos(u);

    double t0 = MPI_Wtime();

    if (method == "jacobi") {
        for (int iter=1; iter<=max_iters; ++iter) {
            for (int i=1;i<=local_ny;++i) {
                for (int j=1;j<=nx;++j) {
                    double up = u[idx(i-1,j,nx_with_ghost)];
                    double down = u[idx(i+1,j,nx_with_ghost)];
                    double left = u[idx(i,j-1,nx_with_ghost)];
                    double right = u[idx(i,j+1,nx_with_ghost)];
                    u_new[idx(i,j,nx_with_ghost)] = ((up + down)*dx2 + (left + right)*dy2)/denom;
                }
            }
            u.swap(u_new);
            exchange_halos(u);

            if ((iter % print_every) == 0 || iter == max_iters) {
                double res = compute_local_residual();
                if (rank==0) std::cout << "Jacobi iter " << iter << " residual = " << std::setprecision(8) << res << "\n";
            }
        }
    } else if (method == "rbgs") {
        for (int iter=1; iter<=max_iters; ++iter) {
            for (int i=1;i<=local_ny;++i) for (int j=1;j<=nx;++j)
                if ((start_row+(i-1)+j)%2==0)
                    u[idx(i,j,nx_with_ghost)] = ((u[idx(i-1,j,nx_with_ghost)]+u[idx(i+1,j,nx_with_ghost)])*dx2
                                                + (u[idx(i,j-1,nx_with_ghost)]+u[idx(i,j+1,nx_with_ghost)])*dy2)/denom;
            exchange_halos(u);
            for (int i=1;i<=local_ny;++i) for (int j=1;j<=nx;++j)
                if ((start_row+(i-1)+j)%2==1)
                    u[idx(i,j,nx_with_ghost)] = ((u[idx(i-1,j,nx_with_ghost)]+u[idx(i+1,j,nx_with_ghost)])*dx2
                                                + (u[idx(i,j-1,nx_with_ghost)]+u[idx(i,j+1,nx_with_ghost)])*dy2)/denom;
            exchange_halos(u);

            if ((iter % print_every) == 0 || iter == max_iters) {
                double res = compute_local_residual();
                if (rank==0) std::cout << "RB-GS iter " << iter << " residual = " << std::setprecision(8) << res << "\n";
            }
        }
    } else {
        if (rank==0) std::cerr << "Unknown method: " << method << "\n";
        MPI_Finalize();
        return 2;
    }

    double t1 = MPI_Wtime();
    double elapsed = t1 - t0;
    double max_elapsed;
    MPI_Reduce(&elapsed, &max_elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    double final_res = compute_local_residual();
    if (rank==0) {
        std::cout << "Final residual = " << final_res << "\n";
        std::cout << "Wall time (max) = " << max_elapsed << " s\n";
    }

    MPI_Finalize();
    return 0;
}