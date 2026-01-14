#include <mpi.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <chrono>
#include <fstream>
#include <sstream>

using namespace std;

struct Body {
    double x, y, z;
    double vx, vy, vz;
    double mass;
};

inline void serialize_block(const vector<Body>& A, vector<double>& buffer) {
    buffer.resize(A.size() * 7);
    for(size_t i = 0; i < A.size(); i++) {
        buffer[i * 7 + 0] = A[i].x;
        buffer[i * 7 + 1] = A[i].y;
        buffer[i * 7 + 2] = A[i].z;
        buffer[i * 7 + 3] = A[i].vx;
        buffer[i * 7 + 4] = A[i].vy;
        buffer[i * 7 + 5] = A[i].vz;
        buffer[i * 7 + 6] = A[i].mass;
    }
}

inline void deserialize_block(vector<Body>& A, const vector<double>& buffer) {
    size_t n = buffer.size() / 7;
    A.resize(n);
    for(size_t i = 0; i < n; i++) {
        A[i].x = buffer[i * 7 + 0];
        A[i].y = buffer[i * 7 + 1];
        A[i].z = buffer[i * 7 + 2];
        A[i].vx = buffer[i * 7 + 3];
        A[i].vy = buffer[i * 7 + 4];
        A[i].vz = buffer[i * 7 + 5];
        A[i].mass = buffer[i * 7 + 6];
    }
}

void init_random(vector<Body>& local, int seed, int64_t start) {
    mt19937_64 rng(seed);
    uniform_real_distribution<double> pos(-1.0, 1.0);
    uniform_real_distribution<double> vel(-0.1, 0.1);
    uniform_real_distribution<double> mass(0.5, 2.0);
    for(size_t i = 0; i < local.size(); i++) {
        local[i].x = pos(rng) + 5.0*sin((start + i) * 0.1);
        local[i].y = pos(rng) + 5.0*cos((start + i) * 0.1);
        local[i].z = pos(rng) * 0.5;
        local[i].vx = vel(rng);
        local[i].vy = vel(rng);
        local[i].vz = vel(rng);
        local[i].mass = mass(rng);
    }
}

void write_checkpoint(const vector<Body>& local, int rank, int step) {
    ostringstream name;
    name << "checkpoint_rank" << rank << "_step" << step << ".bin";
    ofstream out(name.str(), ios::binary);
    size_t n = local.size();
    out.write(reinterpret_cast<const char*>(&n), sizeof(n));
    out.write(reinterpret_cast<const char*>(local.data()), n * sizeof(Body));
}

bool read_checkpoint(vector<Body>& local, int rank, int& step_out) {
    ostringstream name;
    name << "checkpoint_rank" << rank << "_last.bin";
    ifstream in(name.str(), ios::binary);
    if(!in.good()) return false;
    size_t n;
    in.read(reinterpret_cast<char*>(&n), sizeof(n));
    local.resize(n);
    in.read(reinterpret_cast<char*>(local.data()), n * sizeof(Body));
    step_out = 0;
    return true;
}

void write_last_checkpoint(const vector<Body>& local, int rank) {
    ostringstream name;
    name << "checkpoint_rank" << rank << "_last.bin";
    ofstream out(name.str(), ios::binary);
    size_t n = local.size();
    out.write(reinterpret_cast<const char*>(&n), sizeof(n));
    out.write(reinterpret_cast<const char*>(local.data()), n * sizeof(Body));
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if(argc < 5) {
        if (rank == 0) {
            cerr << "Usage: " << argv[0] << " N_total steps dt checkpoint_interval [--restart]" << endl;
        }
        MPI_Finalize();
        return 1;
    }

    int64_t N_total = stoll(argv[1]);
    int steps = stoi(argv[2]);
    double dt = stod(argv[3]);
    int checkpoint_interval = stoi(argv[4]);
    bool restart = false;
    if(argc > 5 && string(argv[5]) == "--restart") {
        restart = true;
    }
    int64_t base = N_total / size;
    int64_t rem = N_total % size;
    int64_t local_n = base + (rank < rem ? 1 : 0);
    int64_t start = rank * base + (rank < rem ? (int64_t)rank : rem);
    vector<Body> local(local_n);

    if(restart) {
        int step_dummy;
        if(!read_checkpoint(local, rank, step_dummy)) {
            if (rank == 0) {
                cerr << "No checkpoint found for rank " << rank << endl;
            }
            init_random(local, 1234 + rank, start);
        }
    } else {
        init_random(local, 1234 + rank, start);
    }

    const double G = 1.0;
    const double soft = 1e-9;
    vector<int> counts(size);
    int mycount = (int)local_n;
    MPI_Allgather(&mycount, 1, MPI_INT, counts.data(), 1, MPI_INT, MPI_COMM_WORLD);
    vector<int> disp(size);
    disp[0] = 0;
    for (int i = 1; i < size; i++) {
        disp[i] = disp[i - 1] + counts[i - 1];
    }

    vector<double> send_buffer;
    serialize_block(local, send_buffer);
    vector<double> recv_buffer;
    recv_buffer.reserve(1024);

    double total_comm_time = 0.0, total_comp_time = 0.0;
    double t_start = MPI_Wtime();
    for(int step = 0; step < steps; step++) {
        double t0 = MPI_Wtime();
        serialize_block(local, send_buffer);
        vector<double> current = send_buffer;
        int src = (rank - 1 + size) % size;
        int tgt = (rank + 1) % size;

        for(int i = 0; i < size - 1; i++) {
            int from_rank = (rank - i - 1 + size) % size;
            int incoming_count = counts[from_rank];
            recv_buffer.resize(incoming_count * 7);
            MPI_Request req_recv, req_send;
            MPI_Irecv(recv_buffer.data(), (int)recv_buffer.size(), MPI_DOUBLE, src, 0, MPI_COMM_WORLD, &req_recv);
            MPI_Isend(current.data(), (int)current.size(), MPI_DOUBLE, tgt, 0, MPI_COMM_WORLD, &req_send);

            double t_comm0 = MPI_Wtime();
            MPI_Wait(&req_recv, MPI_STATUS_IGNORE);
            MPI_Wait(&req_send, MPI_STATUS_IGNORE);
            double t_comm1 = MPI_Wtime();
            total_comm_time += (t_comm1 - t_comm0);

            vector<Body> remote;
            deserialize_block(remote, recv_buffer);
            double t_comp0 = MPI_Wtime();
            for(size_t j = 0; j < local.size(); j++) {
                double ax = 0.0, ay = 0.0, az = 0.0;
                for(size_t k = 0; k < remote.size(); k++) {
                    double dx = remote[k].x - local[j].x;
                    double dy = remote[k].y - local[j].y;
                    double dz = remote[k].z - local[j].z;
                    double dist2 = dx * dx + dy * dy + dz * dz + soft;
                    double invr3 = 1.0 / (sqrt(dist2) * dist2);
                    double f = G * remote[k].mass * invr3;
                    ax += f * dx;
                    ay += f * dy;
                    az += f * dz;
                }
                local[j].vx += ax * dt;
                local[j].vy += ay * dt;
                local[j].vz += az * dt;
            }
            double t_comp1 = MPI_Wtime();
            total_comp_time += (t_comp1 - t_comp0);
            current.swap(recv_buffer);
        }
        double t_comp0 = MPI_Wtime();
        for(size_t j = 0; j < local.size(); j++) {
            double ax = 0.0, ay = 0.0, az = 0.0;
            for(size_t k = 0; k < local.size(); k++) {
                if (j == k) continue;
                double dx = local[k].x - local[j].x;
                double dy = local[k].y - local[j].y;
                double dz = local[k].z - local[j].z;
                double dist2 = dx * dx + dy * dy + dz * dz + soft;
                double invr3 = 1.0 / (sqrt(dist2) * dist2);
                double f = G * local[k].mass * invr3;
                ax += f * dx;
                ay += f * dy;
                az += f * dz;
            }
            local[j].vx += ax * dt;
            local[j].vy += ay * dt;
            local[j].vz += az * dt;
        }
        double t_comp1 = MPI_Wtime();
        total_comp_time += (t_comp1 - t_comp0);

        for(size_t pos = 0; pos < local.size(); pos++) {
            local[pos].x += local[pos].vx * dt;
            local[pos].y += local[pos].vy * dt;
            local[pos].z += local[pos].vz * dt;
        }
        double t1 = MPI_Wtime();
        if ((step + 1) % checkpoint_interval == 0) {
            write_checkpoint(local, rank, step + 1);
            write_last_checkpoint(local, rank);
        } 
        if(rank == 0 && (step % 10 ==0 || step == steps - 1)) {
            double elapsed = MPI_Wtime() - t_start;
            cout << "step " << step << " elapsed(s): " << elapsed << endl;
        }
    }
    double t_end = MPI_Wtime();
    double local_total_time = t_end - t_start;
    double max_total_time;
    MPI_Reduce(&local_total_time, &max_total_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    double sum_comp = total_comp_time;
    double sum_comm = total_comm_time;
    double global_comp_time, global_comm_time;
    MPI_Reduce(&sum_comp, &global_comp_time, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&sum_comm, &global_comm_time, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if(rank == 0) {
        cout << "Max runtime (s): " << max_total_time << endl;
        cout << "Total computation time (s): " << global_comp_time << endl;
        cout << "Total communication time (s): " << global_comm_time << endl;
        if (max_total_time > 0) {
            double efficiency = global_comp_time / (size * max_total_time);
            cout << "Parallel Efficiency: " << efficiency * 100.0 << "%" << endl;
        }
    }

    MPI_Finalize();
    return 0;
}