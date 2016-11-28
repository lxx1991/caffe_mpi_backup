#include <algorithm>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
__global__ void MaskForward(const int nthreads, const Dtype threshold, int spatial_dim, const int channels,
    const Dtype* prob, Dtype* mask) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    mask[index] = 1;
    for (int i=0; i<channels; i++)
      if (prob[i * spatial_dim + index] > threshold + 1e-9) mask[index] = 0;
  }
}


template <typename Dtype>
__global__ void MaskForward(const int nthreads, const Dtype threshold_easy, const Dtype threshold_hard, const int spatial_dim, const int channels,
    const Dtype* prob, Dtype* mask, const int ignore_label, const Dtype* label, Dtype* new_label) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    mask[index] = 1;
    const int label_value = static_cast<int>(label[index]);
    new_label[index] = label_value;
    if (label_value < channels && prob[label_value * spatial_dim + index] > threshold_easy + 1e-9)
    {
      mask[index] = 0;
      new_label[index] = ignore_label;
    }
    else
    {
      for (int i = 0; i<channels; i++)
        if (prob[i * spatial_dim + index] > threshold_hard + 1e-9)
        {
          mask[index] = 0;
          new_label[index] = ignore_label;
        }
    }
  }
}

template <typename Dtype>
void MaskLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

  const Dtype* prob = bottom[0]->gpu_data();
  Dtype* mask = top[0]->mutable_gpu_data();
  const int count = top[0]->count();
  int spatial_dim = count;

  if (top.size() == 1)
  {
     MaskForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, threshold_easy_, spatial_dim, bottom[0]->channels(), prob, mask);
  }
  else
  {
      const Dtype* label_data = bottom[1]->gpu_data();
      Dtype* new_label = top[1]->mutable_gpu_data();

      MaskForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, threshold_easy_, threshold_hard_, spatial_dim, bottom[0]->channels(), prob, mask, ignore_label_, label_data, new_label);
  }

  
  CUDA_POST_KERNEL_CHECK;
}

template <typename Dtype>
void MaskLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom)
{
  
}


INSTANTIATE_LAYER_GPU_FUNCS(MaskLayer);


}  // namespace caffe
