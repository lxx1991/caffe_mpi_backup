#include <vector>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"


namespace caffe {

template <typename Dtype>
void SpatialFilterLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  Layer<Dtype>::LayerSetUp(bottom, top);
}



template <typename Dtype>
void SpatialFilterLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top)
{
  CHECK_EQ(bottom[0]->num(), 1);
  CHECK_EQ(bottom[0]->channels(), 1);
  CHECK_EQ(bottom[0]->num(), bottom[1]->num());
  CHECK_EQ(bottom[0]->height(), bottom[1]->height());
  CHECK_EQ(bottom[0]->width(), bottom[1]->width());
  top[0]->ReshapeLike(*bottom[1]);
}



#ifdef CPU_ONLY
STUB_GPU_FORWARD(SpatialFilterLayer, Forward);
#endif

INSTANTIATE_CLASS(SpatialFilterLayer);
REGISTER_LAYER_CLASS(SpatialFilter);

}  // namespace caffe
