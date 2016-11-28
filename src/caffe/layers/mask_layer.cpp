#include <vector>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"


namespace caffe {

template <typename Dtype>
void MaskLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

  Layer<Dtype>::LayerSetUp(bottom, top);
  
  threshold_easy_ = this->layer_param_.mask_param().threshold_easy();  
  threshold_hard_ = this->layer_param_.mask_param().threshold_hard();  
  
  if (top.size() == 2)
    ignore_label_ = this->layer_param_.mask_param().ignore_label();
}



template <typename Dtype>
void MaskLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top)
{
  CHECK_EQ(bottom[0]->num(), 1);
  top[0]->Reshape(bottom[0]->num(), 1, bottom[0]->height(), bottom[0]->width());
  if (top.size() == 2)
    top[1]->ReshapeLike(*bottom[1]);
}



#ifdef CPU_ONLY
STUB_GPU_FORWARD(MaskLayer, Forward);
#endif

INSTANTIATE_CLASS(MaskLayer);
REGISTER_LAYER_CLASS(Mask);

}  // namespace caffe
