#include <sycl/sycl.hpp>
constexpr int N=400;

int main()
{
	
	sycl::queue q;
        int data[N]; 

	
	for(int i=0; i<N; i++ )
	{
		data[i] = i;
	}

auto data_d = sycl::malloc_device<int>(N, q);
    
    q.memcpy(data_d, data, sizeof(int) * N).wait();
    
    q.parallel_for(N, [=](auto i) { 
      data_d[i] = data_d[i]*data_d[i] ; 
    }).wait();
    
    q.memcpy(data, data_d, sizeof(int) * N).wait();


for(int i=0;i<N;i++)
    {
        std::cout << data[i] << "\t";
    }
    sycl::free(data_d, q);
		
}
