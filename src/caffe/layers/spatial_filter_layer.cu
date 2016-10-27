#include <algorithm>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
__global__ void SpatialFilter(const int nthreads, const int channels, const int spatial_dim, const Dtype* data, const Dtype* mask, Dtype* data_out) {
  CUDA_KERNEL_LOOP(index, nthreads) {
      for (int i=0; i<channels; i++)
      {
        if (mask[index] < 0.5)
          data_out[i * spatial_dim + index] = 0;
        else
          data_out[i * spatial_dim + index] = data[i * spatial_dim + index];
      }
  }
}


template <typename Dtype>
void SpatialFilterLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  
  const Dtype* mask = bottom[0]->gpu_data();
  const Dtype* data = bottom[1]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  const int count = bottom[0]->count();
  const int spatial_dim = count;

  SpatialFilter<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, bottom[1]->channels(), spatial_dim, data, mask, top_data);

  CUDA_POST_KERNEL_CHECK;
}

template <typename Dtype>
void SpatialFilterLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom)
{
  const Dtype* mask = bottom[0]->gpu_data();
  const Dtype* data = top[0]->gpu_diff();
  Dtype* bottom_diff = bottom[1]->mutable_gpu_diff();
  const int count = bottom[0]->count();
  const int spatial_dim = count;

  SpatialFilter<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, bottom[1]->channels(), spatial_dim, data, mask, bottom_diff);
  
  CUDA_POST_KERNEL_CHECK;
}


INSTANTIATE_LAYER_GPU_FUNCS(SpatialFilterLayer);


}  // namespace caffe
