// Runs the actual production collect_starts_kernel, including edge intervals.
#include <cstdio>
#include <vector>
#ifndef CAR_GEOMETRY_SOURCE
#define CAR_GEOMETRY_SOURCE "bevfusion/camera-geometry.cu"
#endif
#include CAR_GEOMETRY_SOURCE
int run_case(const char* name, int total, int remain, std::vector<int> starts, std::vector<int> geometry, int expected_count) {
    int *indices_d, *starts_d, *geometry_d; int3 *out;
    cudaMallocManaged(&indices_d,total*sizeof(int));cudaMallocManaged(&geometry_d,total*sizeof(int));
    cudaMallocManaged(&starts_d,starts.size()*sizeof(int));cudaMallocManaged(&out,starts.size()*sizeof(int3));
    for(int i=0;i<total;i++){indices_d[i]=i;geometry_d[i]=geometry[i];}
    for(int i=0;i<(int)starts.size();i++)starts_d[i]=starts[i];
    bevfusion::camera::collect_starts_kernel<<<1,32>>>(starts.size(),remain,total,indices_d,starts_d,geometry_d,out);
    cudaError_t err=cudaDeviceSynchronize();if(err!=cudaSuccess){printf("CUDA error %s\n",cudaGetErrorString(err));return 1;}
    int count=0;bool last_present=false;bool valid=true;
    for(int i=0;i<(int)starts.size();i++){
        valid &= out[i].x==starts[i]+remain && out[i].z==geometry[starts[i]+remain];
        for(int j=out[i].x;j<out[i].y;j++){count++;if(j==total-1)last_present=true;}
    }
    valid &= count==expected_count && last_present && out[starts.size()-1].y==total;
    printf("%s: covered=%d expected=%d last_point_included=%d end=%d expected_end=%d %s\n",name,count,expected_count,last_present,out[starts.size()-1].y,total,valid?"PASS":"FAIL");
    cudaFree(indices_d);cudaFree(starts_d);cudaFree(geometry_d);cudaFree(out);return valid?0:1;
}
int main(){
 int errors=0;
 errors+=run_case("only_one_projection",1,0,{0},{7},1);
 errors+=run_case("last_cell_single_projection",4,0,{0,3},{4,4,4,9},4);
 errors+=run_case("invalid_prefix_and_last_multi_projection",6,2,{0,2},{0,0,4,4,9,9},4);
 return errors?1:0;
}
