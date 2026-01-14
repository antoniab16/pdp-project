# PDP Project - N-Body Simulation

This project contains implementations of an N-body gravitational simulation using different parallel computing approaches.

## Project Structure

```
pdp_project/
├── thread-based/     # Thread-based implementation
├── mpi/             # MPI (Message Passing Interface) implementation
├── openCL/          # OpenCL implementation
└── README.md        # This file
```

## MPI N-Body Simulation

### Building the Program

To compile the MPI implementation, you need to have an MPI compiler installed (e.g., OpenMPI or MPICH):

```bash
cd mpi
mpicxx -O2 -o nbody_mpi nbody_mpi.cpp -lm
```

### Running the Program

The program requires the following command-line arguments:

```bash
mpirun -np <num_processes> ./nbody_mpi <N_total> <steps> <dt> <checkpoint_interval> [--restart]
```

**Parameters:**
- `<num_processes>`: Number of MPI processes to use
- `<N_total>`: Total number of bodies in the simulation
- `<steps>`: Number of simulation steps to run
- `<dt>`: Time step size (delta time)
- `<checkpoint_interval>`: How often to save checkpoints (in steps)
- `--restart` (optional): Resume from the last checkpoint instead of initializing randomly

### Example Usage

```bash
# Run with 4 processes, 1000 bodies, 100 steps, dt=0.01, checkpoint every 10 steps
mpirun -np 4 ./nbody_mpi 1000 100 0.01 10

# Run with 8 processes and resume from checkpoint
mpirun -np 8 ./nbody_mpi 10000 500 0.01 20 --restart
```

### Output

The program will display:
- Progress updates every 10 steps
- Maximum runtime across all processes
- Total computation time
- Total communication time
- Parallel efficiency percentage

### Checkpointing

The program automatically creates checkpoints at specified intervals, storing the simulation state. Use the `--restart` flag to resume from the last checkpoint instead of reinitializing with random values.
