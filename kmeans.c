#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <time.h>
#include <math.h>
#include <mpich/mpi.h>
#include <CL/cl.h>

// MPI
#define ROOT 0

// OpenCL
#define CL_PROGRAM "kmeans.cl"
#define DIST_FUNC "distances"
#define MEAN_FUNC "means"

// stores point coordinates
typedef struct
{
    cl_int x;
    cl_int y;
    cl_int dist;    // distance to centroid
    cl_int cluster; // cluster coord belongs to
    cl_int is_mean; // bool
} coord;

typedef struct
{
    cl_int length;
    cl_int x;
    cl_int y;
} cluster;

// OpenCL mem
cl_device_id device;
cl_context context;
cl_program program;
cl_kernel dist_kernel;
cl_kernel mean_kernel;
cl_command_queue queue;
cl_event event = NULL;
cl_int err;
cl_mem cl_points;
cl_mem cl_means;
cl_mem cl_cont;
cl_mem cl_sum_x;
cl_mem cl_sum_y;

int distance(coord p1, cluster p2);

// OpenCL
cl_device_id create_device();
cl_program build_program(cl_context ctx, cl_device_id dev, const char *filename);
int cl_freeMem();

int main(int argc, char **argv)
{
    srand(time(NULL));
    if (argc < 4)
    {
        printf("Usage: %s <size> <clusters> <n-points>\n", argv[0]);
        exit(-1);
    }

    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Create custom MPI datatypes
    // coord: int x, int y, int dist, int cluster, int is_mean
    MPI_Datatype MPI_coord;
    MPI_Datatype coord_types[5] = {MPI_INT, MPI_INT, MPI_INT, MPI_INT, MPI_INT};
    int coord_blocklen[5] = {1, 1, 1, 1, 1};
    MPI_Aint coord_disp[] = {offsetof(coord, x),
                             offsetof(coord, y),
                             offsetof(coord, dist),
                             offsetof(coord, cluster),
                             offsetof(coord, is_mean)};
    MPI_Type_create_struct(5, coord_blocklen, coord_disp, coord_types, &MPI_coord);
    MPI_Type_commit(&MPI_coord);

    // cluster: int length, int x, int y
    MPI_Datatype MPI_cluster;
    MPI_Datatype cluster_types[3] = {MPI_INT, MPI_INT, MPI_INT};
    int cluster_blocklen[3] = {1, 1, 1};
    MPI_Aint cluster_disp[] = {offsetof(cluster, length),
                               offsetof(cluster, x),
                               offsetof(cluster, y)};
    MPI_Type_create_struct(3, cluster_blocklen, cluster_disp, cluster_types, &MPI_cluster);
    MPI_Type_commit(&MPI_cluster);

    int SIZE = atoi(argv[1]);
    int K = atoi(argv[2]);
    int num_of_points = atoi(argv[3]);
    int pnt;

    if (num_of_points % size != 0)
    {
        perror("Number of points must be divisible by number of threads.");
        exit(1);
    }

    // Memory allocations
    int *s = (int *)malloc(SIZE * SIZE * sizeof(int));
    coord *points = (coord *)malloc(num_of_points * sizeof(coord));
    cluster *means = (cluster *)malloc(K * sizeof(cluster));
    // OpenCL buffers
    int buffer_size = num_of_points / size;
    coord *points_buf = (coord *)malloc(buffer_size * sizeof(coord));
    int *cont = (int *)malloc(buffer_size * sizeof(int));
    cl_int *sum_x = (cl_int *)malloc(K * sizeof(cl_int));
    cl_int *sum_y = (cl_int *)malloc(K * sizeof(cl_int));
    int *total_sum_x = (int *)malloc(K * sizeof(int));
    int *total_sum_y = (int *)malloc(K * sizeof(int));

    if (rank == ROOT)
    {
        // initialise s
        for (int i = 0; i < SIZE * SIZE; i++)
            s[i] = 0;

        int points_initialised = 0;
        while (points_initialised < num_of_points)
        {
            int offset = (rand() % SIZE) * SIZE + (rand() % SIZE);
            s[offset] = 1;
            points_initialised++;
        }
    }

    if (rank == ROOT)
    {
        // initialise coordinates aray
        pnt = 0;
        for (int i = 0; i < SIZE; i++)
        {
            for (int j = 0; j < SIZE; j++)
            {
                int offset = i * SIZE + j;
                if (s[offset] == 1)
                {
                    points[pnt].y = i;
                    points[pnt].x = j;
                    points[pnt].is_mean = 0;
                    points[pnt].dist = __INT_MAX__;
                    points[pnt].cluster = -1;
                    pnt++;
                }
            }
        }

        // initialise means
        for (int k = 0; k < K; k++)
        {
            int seed_mean;
            do
            {
                seed_mean = rand() % num_of_points;
            } while (points[seed_mean].is_mean == 1);

            means[k].x = points[seed_mean].x;
            means[k].y = points[seed_mean].y;
            means[k].length = 0;
            points[seed_mean].is_mean = 1;
        }
    }

    // begin benchmark
    double start, finish, exec_time;
    if (rank == ROOT)
    {
        start = MPI_Wtime();
    }

    // OpenCL setup
    // Create device and context
    device = create_device();
    context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    if (err < 0)
    {
        perror("Couldn't create a context");
        exit(1);
    }

    // Create command queue
    queue = clCreateCommandQueueWithProperties(context, device, 0, &err);
    if (err < 0)
    {
        perror("Couldn't create a command queue");
        exit(1);
    };

    // Build program & kernels
    program = build_program(context, device, CL_PROGRAM);
    dist_kernel = clCreateKernel(program, DIST_FUNC, &err);
    if (err < 0)
    {
        perror("Couldn't create distances kernel");
        exit(1);
    };

    mean_kernel = clCreateKernel(program, MEAN_FUNC, &err);
    if (err < 0)
    {
        perror("Couldn't create means kernel");
        exit(1);
    };

    /* Each point declares if the loop should continue in the 
    cl kernel. This is then stored in cl_cont, read to cont and
    reduced using a logical OR MPI function into all_cont */
    int all_cont = 1;
    int iterations = 0;
    MPI_Barrier(MPI_COMM_WORLD);
    while (all_cont == 1)
    {
        // initialise cont
        all_cont = 0;
        for (int i = 0; i < buffer_size; i++)
        {
            cont[i] = 0;
        }

        if (rank == ROOT)
        {
            iterations++;
        }

        // MPI: make data available to all threads
        MPI_Scatter(points, buffer_size, MPI_coord, points_buf, buffer_size, MPI_coord, ROOT, MPI_COMM_WORLD);
        MPI_Bcast(means, K, MPI_cluster, ROOT, MPI_COMM_WORLD);

        // OpenCL
        // Create buffers
        cl_points = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, buffer_size * sizeof(coord), points_buf, &err);
        cl_means = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, K * sizeof(cluster), means, &err);
        cl_cont = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, buffer_size * sizeof(cl_int), cont, &err);
        if (err < 0)
        {
            perror("Couldn't create a buffer");
            exit(1);
        };

        // Send buffers to GPU
        err = clEnqueueWriteBuffer(queue, cl_points, CL_TRUE, 0, buffer_size * sizeof(coord), points_buf, 0, NULL, NULL);
        err |= clEnqueueWriteBuffer(queue, cl_means, CL_TRUE, 0, K * sizeof(cluster), means, 0, NULL, NULL);
        err |= clEnqueueWriteBuffer(queue, cl_cont, CL_TRUE, 0, buffer_size * sizeof(cl_int), cont, 0, NULL, NULL);
        if (err < 0)
        {
            perror("Couldn't copy data to GPU/CPU");
            exit(1);
        };

        // Create kernel arguments
        err = clSetKernelArg(dist_kernel, 0, sizeof(cl_mem), (void *)&cl_points);
        err |= clSetKernelArg(dist_kernel, 1, sizeof(cl_mem), (void *)&cl_means);
        err |= clSetKernelArg(dist_kernel, 2, sizeof(cl_int), (void *)&K);
        err |= clSetKernelArg(dist_kernel, 3, sizeof(cl_mem), (void *)&cl_cont);
        if (err < 0)
        {
            perror("Couldn't create a kernel argument");
            exit(1);
        }

        // Enqueue dist kernel
        const size_t global_dist[1] = {buffer_size};
        MPI_Barrier(MPI_COMM_WORLD);
        err = clEnqueueNDRangeKernel(queue, dist_kernel, 1, NULL, global_dist, NULL, 0, NULL, &event);
        err |= clWaitForEvents(1, &event);
        if (err < 0)
        {
            printf("Error: %d -- ", err);
            perror("Couldn't enqueue the kernel");
            exit(1);
        }

        // Read kernel output
        MPI_Barrier(MPI_COMM_WORLD);
        err = clEnqueueReadBuffer(queue, cl_points, CL_TRUE, 0, buffer_size * sizeof(coord), points_buf, 0, NULL, NULL);
        err |= clEnqueueReadBuffer(queue, cl_cont, CL_TRUE, 0, buffer_size * sizeof(cl_int), cont, 0, NULL, NULL);
        if (err < 0)
        {
            perror("Couldn't read the buffer");
            exit(1);
        }

        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Allreduce(cont, &all_cont, 1, MPI_INT, MPI_LOR, MPI_COMM_WORLD);
        MPI_Gather(points_buf, buffer_size, MPI_coord, points, buffer_size, MPI_coord, ROOT, MPI_COMM_WORLD);

        if (rank == ROOT)
        {
            // get size of each cluster
            for (int i = 0; i < num_of_points; i++)
                means[points[i].cluster].length++;
        }

        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Scatter(points, buffer_size, MPI_coord, points_buf, buffer_size, MPI_coord, ROOT, MPI_COMM_WORLD);

        // Mean update
        // OpenCL
        // Create buffers
        cl_points = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, buffer_size * sizeof(coord), points_buf, &err);
        cl_means = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, K * sizeof(cluster), means, &err);
        cl_sum_x = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, K * sizeof(cl_int), sum_x, &err);
        cl_sum_y = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, K * sizeof(cl_int), sum_y, &err);
        if (err < 0)
        {
            perror("Couldn't create a buffer");
            exit(1);
        };

        // Send buffers to GPU
        err = clEnqueueWriteBuffer(queue, cl_points, CL_TRUE, 0, buffer_size * sizeof(coord), points_buf, 0, NULL, NULL);
        err |= clEnqueueWriteBuffer(queue, cl_means, CL_TRUE, 0, K * sizeof(cluster), means, 0, NULL, NULL);
        err |= clEnqueueWriteBuffer(queue, cl_sum_x, CL_TRUE, 0, K * sizeof(cl_int), sum_x, 0, NULL, NULL);
        err |= clEnqueueWriteBuffer(queue, cl_sum_y, CL_TRUE, 0, K * sizeof(cl_int), sum_y, 0, NULL, NULL);
        if (err < 0)
        {
            perror("Couldn't copy data to GPU/CPU");
            exit(1);
        };

        // Create kernel arguments
        err = clSetKernelArg(mean_kernel, 0, sizeof(cl_mem), (void *)&cl_points);
        err |= clSetKernelArg(mean_kernel, 1, sizeof(cl_mem), (void *)&cl_means);
        err |= clSetKernelArg(mean_kernel, 2, sizeof(cl_mem), (void *)&cl_sum_x);
        err |= clSetKernelArg(mean_kernel, 3, sizeof(cl_mem), (void *)&cl_sum_y);
        err |= clSetKernelArg(mean_kernel, 4, sizeof(cl_int), (void *)&buffer_size);
        if (err < 0)
        {
            perror("Couldn't create a mean kernel argument");
            exit(1);
        }

        // Enqueue means kernel
        const size_t global_means[] = {K};
        MPI_Barrier(MPI_COMM_WORLD);
        err = clEnqueueNDRangeKernel(queue, mean_kernel, 1, NULL, global_means, NULL, 0, NULL, &event);
        err |= clWaitForEvents(1, &event);
        if (err < 0)
        {
            printf("Error: %d -- ", err);
            perror("Couldn't enqueue the kernel");
            exit(1);
        }

        // Read kernel output
        MPI_Barrier(MPI_COMM_WORLD);
        err = clEnqueueReadBuffer(queue, cl_sum_x, CL_TRUE, 0, K * sizeof(cl_int), sum_x, 0, NULL, NULL);
        err |= clEnqueueReadBuffer(queue, cl_sum_y, CL_TRUE, 0, K * sizeof(cl_int), sum_y, 0, NULL, NULL);
        if (err < 0)
        {
            perror("Couldn't read the buffer");
            exit(1);
        }

        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Reduce(sum_x, total_sum_x, K, MPI_INT, MPI_SUM, ROOT, MPI_COMM_WORLD);
        MPI_Reduce(sum_y, total_sum_y, K, MPI_INT, MPI_SUM, ROOT, MPI_COMM_WORLD);

        if (rank == ROOT)
        {
            for (int k = 0; k < K; k++)
            {
                means[k].x = total_sum_x[k] / means[k].length;
                means[k].y = total_sum_y[k] / means[k].length;
                means[k].length = 0;
            }
        }

        MPI_Barrier(MPI_COMM_WORLD);
    }

    cl_freeMem();

    // end benchmark
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == ROOT)
    {
        finish = MPI_Wtime();
        exec_time = finish - start;
    }

    MPI_Finalize();
    if (rank == ROOT)
    {
        // print clustered data
        if (SIZE <= 50)
        {
            pnt = 0;
            for (int i = 0; i < SIZE; i++)
            {
                for (int j = 0; j < SIZE; j++)
                {
                    if (points[pnt].x == j && points[pnt].y == i)
                    {
                        printf("%d ", points[pnt].cluster);
                        pnt++;
                    }
                    else
                    {
                        printf("  ");
                    }
                }
                printf("\n");
            }
        }
        printf("Iterations: %d\n", iterations);
        printf("Time: %f\n", exec_time);

        char filename[80];
        sprintf(filename, "./%d %d %d kmeans.txt", SIZE, K, num_of_points);
        FILE *f = fopen(filename, "a");
        if (f == NULL)
        {
            printf("Error: failed to open file.\n");
            exit(1);
        }

        fprintf(f, "%f\n", exec_time);
        fclose(f);
    }

    // frees
    free(s);
    free(points);
    free(means);
    free(points_buf);
    free(cont);
    free(sum_x);
    free(sum_y);
    free(total_sum_x);
    free(total_sum_y);

    return 0;
}

// calculate squared euclidean distance between a point and a mean
int distance(coord p1, cluster p2)
{
    int x = pow(p2.x - p1.x, 2);
    int y = pow(p2.y - p1.y, 2);
    return x + y;
}

// Find a GPU or CPU to use
cl_device_id create_device()
{

    cl_platform_id platform;
    cl_device_id dev;
    int err;

    // ID plaform
    err = clGetPlatformIDs(1, &platform, NULL);
    if (err < 0)
    {
        perror("Platform ID not retrieved");
        exit(1);
    }

    // Access device
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &dev, NULL);
    if (err == CL_DEVICE_NOT_FOUND)
    {
        err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 1, &dev, NULL);
    }
    if (err < 0)
    {
        perror("Device not found");
        exit(1);
    }

    return dev;
}

cl_program build_program(cl_context ctx, cl_device_id dev, const char *filename)
{
    cl_program program;
    FILE *program_handle;
    char *program_buffer, *program_log;
    size_t program_size, log_size;
    int err;

    // Read and insert program file into buffer
    program_handle = fopen(filename, "r");
    if (program_handle == NULL)
    {
        perror("Program file could not be opened");
        exit(1);
    }
    fseek(program_handle, 0, SEEK_END);
    program_size = ftell(program_handle);
    rewind(program_handle);
    program_buffer = (char *)malloc(program_size + 1);
    program_buffer[program_size] = '\0';
    fread(program_buffer, sizeof(char), program_size, program_handle);
    fclose(program_handle);

    // Create CL program
    program = clCreateProgramWithSource(ctx, 1, (const char **)&program_buffer, &program_size, &err);
    if (err < 0)
    {
        perror("Couldn't create program");
        exit(1);
    }
    free(program_buffer);

    // Build program
    err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    if (err < 0)
    {
        // print log
        clGetProgramBuildInfo(program, dev, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        program_log = (char *)malloc(log_size + 1);
        program_log[log_size] = '\0';
        clGetProgramBuildInfo(program, dev, CL_PROGRAM_BUILD_LOG, log_size + 1, program_log, NULL);
        printf("%s\n", program_log);
        free(program_log);
        exit(1);
    }

    return program;
}

int cl_freeMem()
{
    int err;
    // free OpenCL memory
    err = clReleaseKernel(dist_kernel);
    err |= clReleaseKernel(mean_kernel);
    err |= clReleaseMemObject(cl_points);
    err |= clReleaseMemObject(cl_means);
    err |= clReleaseMemObject(cl_cont);
    err |= clReleaseMemObject(cl_sum_x);
    err |= clReleaseMemObject(cl_sum_y);
    err |= clReleaseCommandQueue(queue);
    err |= clReleaseProgram(program);
    err |= clReleaseContext(context);
    return err;
}
