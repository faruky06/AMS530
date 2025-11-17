#include <mpi.h>
#include <random>
#include <vector>
#include <cmath>
#include <iostream>
#include <algorithm>
#include <chrono>
#include <array>
#include <iomanip>

#define N (1 << 16)

template<typename T>
void print_vec(const std::vector<T>& vec){
    std::for_each(vec.begin(),
                  vec.end(),
                  [](const auto& i){ std::cout << i << " "; });
}

void print_arr(const std::array<double, 3>& arr){
    std::for_each(arr.begin(), arr.end(), [](const auto& i){ std::cout << i << " "; });
}

int main(int argc, char **argv){
    MPI_Init(&argc, &argv);

    int rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    if (N % world_size != 0) {
        if (rank == 0)
            std::cerr << "Number of processors must divide 2^16!\n";
        MPI_Finalize();
        return 1;
    }

    int sendcount = N / world_size;

    // MPI datatype for a particle position (x,y,z)
    MPI_Datatype MPI_PARTICLE;
    MPI_Type_contiguous(3, MPI_DOUBLE, &MPI_PARTICLE);
    MPI_Type_commit(&MPI_PARTICLE);

    // RNG per rank
    std::mt19937_64 rng(rank + 1); // avoid zero seed
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    std::gamma_distribution<double> dist_2(3.0, 100.0);

    // Each rank generates its own block of 'sendcount' particle coordinates
    std::vector<std::array<double, 3>> local_recv(sendcount);

    for (int i = 0; i < sendcount; ++i) {
        double r;
        do {
            r = dist_2(rng);
        } while (r > 100.0);

        double phi = (2.0 * M_PI) * dist(rng);
        double arg = cbrt((2.0 * dist(rng)) - 1.0);
        arg = std::clamp(arg, -1.0, 1.0);
        double theta = acos(arg);

        local_recv[i] = { r * sin(theta) * cos(phi),
                          r * sin(theta) * sin(phi),
                          r * cos(theta) };
    }

    // Allgather so every rank has the full array of particle positions
    std::vector<std::array<double, 3>> particles(N);
    MPI_Allgather(local_recv.data(), sendcount, MPI_PARTICLE,
                  particles.data(), sendcount, MPI_PARTICLE,
                  MPI_COMM_WORLD);

    //container for local forces (only stores full N on root if gathered)
    // compute forces for indices
    int start_idx = rank * sendcount;
    int end_idx = start_idx + sendcount;

    std::vector<std::array<double, 3>> local_forces(sendcount);
    for (int i = 0; i < sendcount; ++i) local_forces[i] = {0.0, 0.0, 0.0};

    const double rc = 10.0;
    const double rc2 = rc * rc;

    MPI_Barrier(MPI_COMM_WORLD);
    auto t0 = std::chrono::high_resolution_clock::now();

    // Compute forces: each rank calculates F_i for its assigned i's by summing over all j != i
    for (int ii = start_idx; ii < end_idx; ++ii) {
        std::array<double, 3> Fi = {0.0, 0.0, 0.0};
        const auto &xi = particles[ii];

        for (int j = 0; j < N; ++j) {
            if (j == ii) continue;
            const auto &xj = particles[j];

            // r_vec = xi - xj
            double rx = xi[0] - xj[0];
            double ry = xi[1] - xj[1];
            double rz = xi[2] - xj[2];

            double r2 = rx*rx + ry*ry + rz*rz;
            if (r2 > rc2) continue; // truncated force
            if (r2 == 0.0) continue; // guard (coincident particles)

            // compute inverse powers 
            double inv_r2 = 1.0 / r2;
            double inv_r6 = inv_r2 * inv_r2 * inv_r2;
            double inv_r12 = inv_r6 * inv_r6;
            double inv_r = std::sqrt(inv_r2);

            //scalar = (1/r^12 - 2/r^6)
            double scalar = (inv_r12 - 2.0 * inv_r6);

            //f_ij = scalar * r_hat = scalar * (r_vec * inv_r)
            double factor = scalar * inv_r;

            Fi[0] += factor * rx;
            Fi[1] += factor * ry;
            Fi[2] += factor * rz;
        }

        // store in local_forces array (index ii - start_idx)
        local_forces[ii - start_idx] = Fi;
    }

    MPI_Barrier(MPI_COMM_WORLD);
    auto t1 = std::chrono::high_resolution_clock::now();
    double local_seconds = std::chrono::duration<double>(t1 - t0).count();

    //Reduce to find max time across ranks
    double max_seconds;
    MPI_Reduce(&local_seconds, &max_seconds, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    //Gather local_forces to root 
    std::vector<std::array<double,3>> global_forces;
    if (rank == 0) global_forces.resize(N);
    MPI_Gather(local_forces.data(), sendcount, MPI_PARTICLE,
    global_forces.data(), sendcount, MPI_PARTICLE,
    0, MPI_COMM_WORLD);

    if (rank == 0) {
        std::cout << std::fixed << std::setprecision(6);
        std::cout << "Force computation complete. Max compute time across ranks: " << max_seconds << " s\n";
        std::cout << "First 10 particle positions and forces (pos_x pos_y pos_z) -> (Fx Fy Fz):\n";
        for (int i = 0; i < std::min(10, N); ++i) {
            auto &p = particles[i];
            auto &f = global_forces[i];
            std::cout << "p[" << i << "]: (" << p[0] << " " << p[1] << " " << p[2] << ") -> ("
                      << f[0] << " " << f[1] << " " << f[2] << ")\n";
        }
    }

    MPI_Type_free(&MPI_PARTICLE);
    MPI_Finalize();
    return 0;
}
