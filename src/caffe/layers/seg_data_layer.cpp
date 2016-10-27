#include <fstream>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include "caffe/data_layers.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

#ifdef USE_MPI
#include "mpi.h"
#include <boost/filesystem.hpp>
using namespace boost::filesystem;
#endif

namespace caffe{
template <typename Dtype>
SegDataLayer<Dtype>:: ~SegDataLayer<Dtype>(){
	this->JoinPrefetchThread();
}

template <typename Dtype>
void SegDataLayer<Dtype>:: DataLayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top){

	const int stride = this->layer_param_.transform_param().stride();
	const string& source = this->layer_param_.seg_data_param().source();
	const string& root_dir = this->layer_param_.seg_data_param().root_dir();


	LOG(INFO) << "Opening file: " << source;
	std:: ifstream infile(source.c_str());
	string img_filename;
	string label_filename;
	while (infile >> img_filename >> label_filename){
		lines_.push_back(std::make_pair(root_dir + img_filename, root_dir + label_filename));
	}

	if (this->layer_param_.seg_data_param().shuffle()){
		const unsigned int prefectch_rng_seed = 17;//caffe_rng_rand(); // magic number
		prefetch_rng_.reset(new Caffe::RNG(prefectch_rng_seed));
		ShuffleImages();
	}

	LOG(INFO) << "A total of " << lines_.size() << " images.";
	lines_id_ = 0;

	Datum datum_data, datum_label;
	CHECK(ReadSegDataToDatum(lines_[lines_id_].first, lines_[lines_id_].second, &datum_data, &datum_label, true));


	int crop_height = datum_data.height() / stride * stride;
	int crop_width = datum_data.width() / stride * stride;


	top[0]->Reshape(1, datum_data.channels(), crop_height, crop_width);
	this->prefetch_data_.Reshape(1, datum_data.channels(), crop_height, crop_width);

	top[1]->Reshape(1, datum_label.channels(), crop_height, crop_width);
	this->prefetch_label_.Reshape(1, datum_label.channels(), crop_height, crop_width);

	LOG(INFO) << "output data size: " << top[0]->num() << "," << top[0]->channels() << "," << top[0]->height() << "," << top[0]->width();
	LOG(INFO) << "output label size: " << top[1]->num() << "," << top[1]->channels() << "," << top[1]->height() << "," << top[1]->width();
}

template <typename Dtype>
void SegDataLayer<Dtype>::ShuffleImages(){
	caffe::rng_t* prefetch_rng = static_cast<caffe::rng_t*>(prefetch_rng_->generator());
	shuffle(lines_.begin(), lines_.end(), prefetch_rng);
}

template <typename Dtype>
void SegDataLayer<Dtype>::InternalThreadEntry(){

	Datum datum_data, datum_label;
	CHECK(this->prefetch_data_.count());
	
	const int lines_size = lines_.size();

	CHECK_GT(lines_size, lines_id_);
	CHECK(ReadSegDataToDatum(lines_[lines_id_].first, lines_[lines_id_].second, &datum_data, &datum_label, true));

	this->data_transformer_->Transform(datum_data, datum_label, &this->prefetch_data_, &this->prefetch_label_);

	//next iteration
	lines_id_++;
	if (lines_id_ >= lines_.size()) {
		// We have reached the end. Restart from the first.
		DLOG(INFO) << "Restarting data prefetching from start.";
		lines_id_ = 0;
		if (this->layer_param_.seg_data_param().shuffle()) {
			ShuffleImages();
		}
	}
}

INSTANTIATE_CLASS(SegDataLayer);
REGISTER_LAYER_CLASS(SegData);
}
