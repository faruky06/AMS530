#include <mpi.h>
#include <vector>
#include <random>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <cstdlib>

inline int idx(int r, int c, int N) { 
    return r * N + c; 
    } //inline for optimization purposes.. store entire matrix in one vector 

void fill_random(std::vector<double>& M, int N, unsigned seed) {
    std::uniform_real_distribution<double> dist(-1, 1);
    std::random_device rng;
    for (int i = 0; i < N * N; ++i) M[i] = dist(rng);
}
void serial_mul(const std::vector<double>& A, const std::vector<double>& B, std::vector<double>& C, int N) {
    for (int i = 0; i < N; ++i) {
        for (int k = 0; k < N; ++k) {
            double a = A[idx(i,k,N)];
            int start = idx(i,0,N);
            int start_2 = idx(k,0,N);
            for (int j = 0; j < N; ++j) {
                C[start + j] += a * B[start_2 + j];
            }
        }
    }
}
//O(n^3) standard serial matrix multiplication
void local_matmul(const std::vector<double>& Alocal, const std::vector<double>& B, std::vector<double>& Clocal, int nlocal, int N) {
    std::fill(Clocal.begin(), Clocal.end(), 0.0);
    for (int i = 0; i < nlocal; ++i) {
        for (int k = 0; k < N; ++k) {
            double a = Alocal[i*N + k];
            int start = i*N;
            int start_2 = k*N;
            for (int j = 0; j < N; ++j) {
                Clocal[start + j] += a * B[start_2 + j];
            }
        }
    }
}

double max_abs_diff(const std::vector<double>& X, const std::vector<double>& Y) {
    double m = 0.0;
    int n = X.size();
    for (int i = 0; i < n; ++i) {
        double d = std::abs(X[i] - Y[i]);
        if (d > m) m = d;
    }
    return m;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc < 2) {
        if (rank == 0) std::cerr << "Usage: " << argv[0] << " N\n";
        MPI_Finalize();
        return 1;
    }
    int N = static_cast<int>(std::stoul(argv[1])); //assumes input is of correct int type.. 
    int nlocal = N / size;

    std::vector<double> A, B, C, C_serial;
    if (rank == 0) {
        try {
            A.assign(N*N, 0.0);
            B.assign(N*N, 0.0);
            C.assign(N*N, 0.0);
            C_serial.assign(N*N, 0.0);
        } catch (std::bad_alloc&) {
            std::cerr << "Root: allocation failed for N = " << N << "\n";
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        unsigned seed = 1234u;
        fill_random(A, N, seed);
        fill_random(B, N, seed+1);
    }
 
    std::vector<double> A_local(nlocal*N);
    std::vector<double> C_local(nlocal*N);
    std::vector<double> B_full(N*N);

    //Scatter rows of A
    MPI_Scatter(rank==0 ? A.data() : nullptr, nlocal*N, MPI_DOUBLE,
                A_local.data(), nlocal*N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    //Broadcast B
    MPI_Bcast(rank==0 ? B.data() : B_full.data(), N*N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    //Copy B into B_full on root process
    if (rank == 0) std::copy(B.begin(), B.end(), B_full.begin());

    //Serial multiplication on root process 
    double t_serial = 0.0;
    if (rank == 0) {
        std::fill(C_serial.begin(), C_serial.end(), 0.0);
        double t0 = MPI_Wtime();
        serial_mul(A, B_full, C_serial, N);
        double t1 = MPI_Wtime();
        t_serial = t1 - t0;
    }
    MPI_Barrier(MPI_COMM_WORLD);
    double t0p = MPI_Wtime();
    //local parallel multiplication
    local_matmul(A_local, B_full, C_local, nlocal, N);
    //the results
    if (rank == 0) std::fill(C.begin(), C.end(), 0.0);
    MPI_Gather(C_local.data(), nlocal*N, MPI_DOUBLE,
               rank==0 ? C.data() : nullptr, nlocal*N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    double t1p = MPI_Wtime();
    double t_parallel = t1p - t0p; //calculate 
    if (rank == 0) {
        double maxerr = max_abs_diff(C, C_serial);
        const double tol = 1e-9;
        bool ok = (maxerr <= tol) || (maxerr / (1.0 + std::abs(C_serial[0])) <= 1e-8); //makes sure that both matrices are the same
        std::cout << std::fixed << std::setprecision(6);
        std::cout << "N = " << N << ", P = " << size << ", rows per process = " << nlocal << "\n";
        std::cout << "serial time:   " << t_serial << " s\n";
        std::cout << "parallel time: " << t_parallel << " s\n";
        if (t_parallel > 0.0)
            std::cout << "speedup:       " << (t_serial / t_parallel) << " (serial/par)\n";
        else
            std::cout << "speedup:       INF (parallel time too small)\n";
        std::cout << "max absolute error: " << maxerr << "\n";
        std::cout << "verification: " << (ok ? "pass" : "fail") << "\n";
    }
    MPI_Finalize();
}