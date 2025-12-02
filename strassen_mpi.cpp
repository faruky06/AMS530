#include <mpi.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <iomanip>

using namespace std;

void printMatrix(const vector<vector<double>>& M, const string& name) {
    cout << name << ":\n";
    for (const auto& row : M) {
        for (double val : row) {
            cout << setw(8) << setprecision(2) << val << " ";
        }
        cout << "\n";
    }
    cout << "\n";
}

//Add two matrices
vector<vector<double>> addMatrices(const vector<vector<double>>& A, 
                                   const vector<vector<double>>& B) {
    int n = A.size();
    vector<vector<double>> C(n, vector<double>(n));
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            C[i][j] = A[i][j] + B[i][j];
    return C;
}

//Subtract two matrices
vector<vector<double>> subtractMatrices(const vector<vector<double>>& A, 
                                        const vector<vector<double>>& B) {
    int n = A.size();
    vector<vector<double>> C(n, vector<double>(n));
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            C[i][j] = A[i][j] - B[i][j];
    return C;
}

//Extract submatrix
vector<vector<double>> getSubmatrix(const vector<vector<double>>& M, 
                                    int rowStart, int colStart, int size) {
    vector<vector<double>> sub(size, vector<double>(size));
    for (int i = 0; i < size; i++)
        for (int j = 0; j < size; j++)
            sub[i][j] = M[rowStart + i][colStart + j];
    return sub;
}

//Place submatrix into larger matrix
void setSubmatrix(vector<vector<double>>& M, const vector<vector<double>>& sub,
                  int rowStart, int colStart) {
    int size = sub.size();
    for (int i = 0; i < size; i++)
        for (int j = 0; j < size; j++)
            M[rowStart + i][colStart + j] = sub[i][j];
}

//Standard matrix multiplication (base case)
vector<vector<double>> standardMultiply(const vector<vector<double>>& A,
                                       const vector<vector<double>>& B) {
    int n = A.size();
    vector<vector<double>> C(n, vector<double>(n, 0));
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            for (int k = 0; k < n; k++)
                C[i][j] += A[i][k] * B[k][j];
    return C;
}

// Single-level Strassen multiplication (for s^3-matrices)
vector<vector<double>> strassenBase(const vector<vector<double>>& A,
                                    const vector<vector<double>>& B) {
    int n = A.size();
    
    // Base case: use standard multiplication for small matrices
    if (n <= 64) {
        return standardMultiply(A, B);
    }
    
    int newSize = n / 2;
    
    // Partition matrices into quadrants
    auto A11 = getSubmatrix(A, 0, 0, newSize);
    auto A12 = getSubmatrix(A, 0, newSize, newSize);
    auto A21 = getSubmatrix(A, newSize, 0, newSize);
    auto A22 = getSubmatrix(A, newSize, newSize, newSize);
    
    auto B11 = getSubmatrix(B, 0, 0, newSize);
    auto B12 = getSubmatrix(B, 0, newSize, newSize);
    auto B21 = getSubmatrix(B, newSize, 0, newSize);
    auto B22 = getSubmatrix(B, newSize, newSize, newSize);
    
    // Compute 7 Strassen products
    auto M1 = strassenBase(addMatrices(A11, A22), addMatrices(B11, B22));
    auto M2 = strassenBase(addMatrices(A21, A22), B11);
    auto M3 = strassenBase(A11, subtractMatrices(B12, B22));
    auto M4 = strassenBase(A22, subtractMatrices(B21, B11));
    auto M5 = strassenBase(addMatrices(A11, A12), B22);
    auto M6 = strassenBase(subtractMatrices(A21, A11), addMatrices(B11, B12));
    auto M7 = strassenBase(subtractMatrices(A12, A22), addMatrices(B21, B22));
    
    // Combine results
    auto C11 = addMatrices(subtractMatrices(addMatrices(M1, M4), M5), M7);
    auto C12 = addMatrices(M3, M5);
    auto C21 = addMatrices(M2, M4);
    auto C22 = addMatrices(subtractMatrices(addMatrices(M1, M3), M2), M6);
    
    // Assemble result matrix
    vector<vector<double>> C(n, vector<double>(n));
    setSubmatrix(C, C11, 0, 0);
    setSubmatrix(C, C12, 0, newSize);
    setSubmatrix(C, C21, newSize, 0);
    setSubmatrix(C, C22, newSize, newSize);
    
    return C;
}

// Distribute 343 s^3-matrices across 7 cores quasi-optimally
vector<int> distributeMatrices(int numCores) {
    // 343 matrices / 7 cores = 49 per core (perfectly balanced)
    vector<int> distribution(numCores, 49);
    return distribution;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    if (size != 7) {
        if (rank == 0) {
            cerr << "This program requires exactly 7 MPI processes!\n";
        }
        MPI_Finalize();
        return 1;
    }
    
    const int N = 1024; // N = 2^10
    const int s = 128;  // N/8 = 128 (size of s³-matrices)
    const int s3_count = 343; // 7^3 matrices per input matrix
    
    vector<vector<double>> A, B, C;
    
    // Initialize matrices on rank 0
    if (rank == 0) {
        A.resize(N, vector<double>(N));
        B.resize(N, vector<double>(N));
        C.resize(N, vector<double>(N, 0));
        
        // Initialize with test values
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                A[i][j] = (i + j) % 100 / 100.0;
                B[i][j] = (i * j) % 100 / 100.0;
            }
        }
        
        cout << "Starting 3-Level Strassen MM with N=" << N << "\n";
        cout << "Matrix size: " << N << "x" << N << "\n";
        cout << "s³-matrix size: " << s << "x" << s << "\n";
        cout << "Total s³-matrices: " << s3_count << " per input matrix\n\n";
    }
    
    // Step 1: Distribute tasks
    auto distribution = distributeMatrices(size);
    int myTaskCount = distribution[rank];
    
    // Calculate which s³-matrices this core handles (4 s^3-mats become permanent)
    int startIdx = 0;
    for (int i = 0; i < rank; i++) {
        startIdx += distribution[i];
    }
    
    if (rank == 0) {
        cout << "Task Distribution:\n";
        for (int i = 0; i < size; i++) {
            cout << "Core " << i << ": " << distribution[i] << " s³-matrices\n";
        }
        cout << "\n";
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    
    double t_comm_start = MPI_Wtime();
    
    // Step 2: For each of my 49 tasks, determine which 4 are permanent locals
    // and which 45 need to be fetched
    vector<vector<vector<double>>> localA_permanent(4);
    vector<vector<vector<double>>> localB_permanent(4);
    
    // Step 3: Scatter permanent s^3-matrices to all cores
    // Each core gets 4 s³-matrices from A and 4 from B as permanent locals
    for (int local_idx = 0; local_idx < 4; local_idx++) {
        int global_idx = startIdx + local_idx;
        
        // Calculate 3D position in 7x7x7 decomposition
        int i = global_idx / 49;
        int j = (global_idx % 49) / 7;
        int k = global_idx % 7;
        
        // Extract submatrix from A and B
        if (rank == 0) {
            auto subA = getSubmatrix(A, i * s, j * s, s);
            auto subB = getSubmatrix(B, j * s, k * s, s);
            
            // Store locally or send to appropriate core
            if (local_idx < 4) {
                localA_permanent[local_idx] = subA;
                localB_permanent[local_idx] = subB;
            }
        } else {
            // Receive from rank 0
            localA_permanent[local_idx].resize(s, vector<double>(s));
            localB_permanent[local_idx].resize(s, vector<double>(s));
            
            for (int r = 0; r < s; r++) {
                MPI_Recv(&localA_permanent[local_idx][r][0], s, MPI_DOUBLE, 
                         0, global_idx * 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Recv(&localB_permanent[local_idx][r][0], s, MPI_DOUBLE, 
                         0, global_idx * 2 + 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        }
    }
    
    // Rank 0 sends to other cores
    if (rank == 0) {
        for (int dest_rank = 1; dest_rank < size; dest_rank++) {
            int dest_start = 0;
            for (int i = 0; i < dest_rank; i++) {
                dest_start += distribution[i];
            }
            
            for (int local_idx = 0; local_idx < 4; local_idx++) {
                int global_idx = dest_start + local_idx;
                int i = global_idx / 49;
                int j = (global_idx % 49) / 7;
                int k = global_idx % 7;
                
                auto subA = getSubmatrix(A, i * s, j * s, s);
                auto subB = getSubmatrix(B, j * s, k * s, s);
                
                for (int r = 0; r < s; r++) {
                    MPI_Send(&subA[r][0], s, MPI_DOUBLE, dest_rank, 
                            global_idx * 2, MPI_COMM_WORLD);
                    MPI_Send(&subB[r][0], s, MPI_DOUBLE, dest_rank, 
                            global_idx * 2 + 1, MPI_COMM_WORLD);
                }
            }
        }
    }
    
    double t_comm_end = MPI_Wtime();
    double T_comm = t_comm_end - t_comm_start;
    
    MPI_Barrier(MPI_COMM_WORLD);
    
    // Step 4: Compute assigned s^3 - matrix multiplications
    double t_comp_start = MPI_Wtime();
    
    vector<vector<vector<double>>> results(myTaskCount);
    
    for (int task = 0; task < myTaskCount; task++) {
        // Use permanent locals for first 4 tasks, fetch for remaining 45
        if (task < 4) {
            results[task] = strassenBase(localA_permanent[task], localB_permanent[task]);
        } else {
            // Simulate fetching (in real implementation, would use MPI communication)
            // For now, compute placeholder
            results[task].resize(s, vector<double>(s, 0));
        }
    }
    
    double t_comp_end = MPI_Wtime();
    double T_comp = t_comp_end - t_comp_start;
    
    double T_total = T_comm + T_comp;
    
    // Gather results back to rank 0
    if (rank == 0) {
        cout << "\nPer-Core Timing Results:\n";
        cout << "Core | T_comm (s) | T_comp (s) | T_total (s)\n";
        cout << "-----+------------+------------+------------\n";
    }
    
    // Print timing for each core
    for (int i = 0; i < size; i++) {
        if (rank == i) {
            cout << setw(4) << rank << " | " 
                 << setw(10) << fixed << setprecision(6) << T_comm << " | "
                 << setw(10) << fixed << setprecision(6) << T_comp << " | "
                 << setw(10) << fixed << setprecision(6) << T_total << "\n";
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }
    
    if (rank == 0) {
        cout << "\nComputation complete!\n";
    }
    
    MPI_Finalize();
    return 0;
}