typedef struct
{
    int x;
    int y;
    int dist;       // distance to centroid
    int cluster;    // cluster coord belongs to
    int is_mean;    // bool
} coord;

typedef struct
{
    int length;
    int x;
    int y;
} cluster;

__kernel void distances(__global coord* points, __global cluster* means,
                          const int K, __global int* cont) {
    const int gl_point = get_global_id(0);    // current point in points array
    int x, y, d;
    for (int k = 0; k < K; k++) {
        x = pow((float)(means[k].x - points[gl_point].x), 2);
        y = pow((float)(means[k].y - points[gl_point].y), 2);
        d = x + y;
        if (d < points[gl_point].dist) {
            points[gl_point].dist = d;
            points[gl_point].cluster = k;
            cont[gl_point] = 1;
        } 
    }
}

__kernel void means(__global coord* points, __global cluster* means,
                    __global int* sum_x, __global int* sum_y,
                    const int buffer_size) {
    const int gl_cluster = get_global_id(0);    // current cluster
    sum_x[gl_cluster] = 0;
    sum_y[gl_cluster] = 0;
    for (int i = 0; i < buffer_size; i++) {
        if (points[i].cluster == gl_cluster) {
            sum_x[gl_cluster] += points[i].x;
            sum_y[gl_cluster] += points[i].y;
        } 
    }
}
