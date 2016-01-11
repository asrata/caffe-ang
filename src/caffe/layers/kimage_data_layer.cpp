#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>

#include <fstream>  // NOLINT(readability/streams)
#include <iostream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>

#include "caffe/data_layers.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

namespace caffe {

template <typename Dtype>
KImageDataLayer<Dtype>::~KImageDataLayer<Dtype>() {
  this->StopInternalThread();
}

template <typename Dtype>
void KImageDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const int new_height = this->layer_param_.kimage_data_param().new_height();
  const int new_width  = this->layer_param_.kimage_data_param().new_width();
  const bool is_color  = this->layer_param_.kimage_data_param().is_color();
  string root_folder = this->layer_param_.kimage_data_param().root_folder();
  const int num_images = this->layer_param_.kimage_data_param().num_images();

  CHECK((new_height == 0 && new_width == 0) ||
      (new_height > 0 && new_width > 0)) << "Current implementation requires "
      "new_height and new_width to be set at the same time.";
  // Read the file with filenames and labels
  const string& source = this->layer_param_.kimage_data_param().source();
  LOG(INFO) << "Opening file " << source;
  std::ifstream infile(source.c_str());
  string filename;
  int label;
  while (infile >> filename >> label) {
    lines_.push_back(std::make_pair(filename, label));
	if (label_index_.count(label) == 0) {
		label_index_[label] = imagesets_.size();
		imagesets_.push_back(make_pair(label, vector<std::string>()));
	}
	imagesets_[label_index_[label]].second.push_back(filename);
  }

  if (this->layer_param_.kimage_data_param().shuffle()) {
    // randomly shuffle data
    LOG(INFO) << "Shuffling data";
    const unsigned int prefetch_rng_seed = caffe_rng_rand();
    prefetch_rng_.reset(new Caffe::RNG(prefetch_rng_seed));
    ShuffleImages();
  }
  LOG(INFO) << "A total of " << lines_.size() << " images.";
  LOG(INFO) << "A total of " << imagesets_.size() << " image sets.";

  imagesets_id_ = 0;
  // Check if we would need to randomly skip a few data points
  if (this->layer_param_.kimage_data_param().rand_skip()) {
    unsigned int skip = caffe_rng_rand() %
        this->layer_param_.kimage_data_param().rand_skip();
    LOG(INFO) << "Skipping first " << skip << " data points.";
    CHECK_GT(imagesets_.size(), skip) << "Not enough points to skip";
    imagesets_id_ = skip;
  }
  // Read an image, and use it to initialize the top blob.
  cv::Mat cv_img = ReadImageToCVMat(root_folder + imagesets_[imagesets_id_].second[0],
                                    new_height, new_width, is_color);
  CHECK(cv_img.data) << "Could not load " << imagesets_[imagesets_id_].second[0];
  // Use data_transformer to infer the expected blob shape from a cv_image.
  vector<int> top_shape = this->data_transformer_->InferBlobShape(cv_img);
  this->transformed_data_.Reshape(top_shape);
  // Reshape prefetch_data and top[0] according to the batch_size.
  const int batch_size = this->layer_param_.kimage_data_param().batch_size();
  CHECK_GT(batch_size, 0) << "Positive batch size required";
  top_shape[0] = batch_size;
  top_shape[1] *= num_images;
  for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
    this->prefetch_[i].data_.Reshape(top_shape);
  }
  top[0]->Reshape(top_shape);

  LOG(INFO) << "output data size: " << top[0]->num() << ","
      << top[0]->channels() << "," << top[0]->height() << ","
      << top[0]->width();
  // label
  vector<int> label_shape(1, batch_size);
  top[1]->Reshape(label_shape);
  for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
    this->prefetch_[i].label_.Reshape(label_shape);
  }
}

template <typename Dtype>
void KImageDataLayer<Dtype>::ShuffleImages() {
  caffe::rng_t* prefetch_rng =
      static_cast<caffe::rng_t*>(prefetch_rng_->generator());
  shuffle(imagesets_.begin(), imagesets_.end(), prefetch_rng);
}

// This function is called on prefetch thread
template <typename Dtype>
void KImageDataLayer<Dtype>::load_batch(Batch<Dtype>* batch) {
  CPUTimer batch_timer;
  batch_timer.Start();
  double read_time = 0;
  double trans_time = 0;
  CPUTimer timer;
  CHECK(batch->data_.count());
  CHECK(this->transformed_data_.count());
  KImageDataParameter kimage_data_param = this->layer_param_.kimage_data_param();
  const int batch_size = kimage_data_param.batch_size();
  const int new_height = kimage_data_param.new_height();
  const int new_width = kimage_data_param.new_width();
  const bool is_color = kimage_data_param.is_color();
  string root_folder = kimage_data_param.root_folder();
  const int num_images = kimage_data_param.num_images();

  // Reshape according to the first image of each batch
  // on single input batches allows for inputs of varying dimension.
  cv::Mat cv_img = ReadImageToCVMat(root_folder + imagesets_[imagesets_id_].second[0],
      new_height, new_width, is_color);
  CHECK(cv_img.data) << "Could not load " << imagesets_[imagesets_id_].second[0];
  // Use data_transformer to infer the expected blob shape from a cv_img.
  vector<int> top_shape = this->data_transformer_->InferBlobShape(cv_img);
  this->transformed_data_.Reshape(top_shape);
  // Reshape batch according to the batch_size.
  top_shape[0] = batch_size * num_images;
  batch->data_.Reshape(top_shape);

  Dtype* prefetch_data = batch->data_.mutable_cpu_data();
  Dtype* prefetch_label = batch->label_.mutable_cpu_data();

  // datum scales
  const int imagesets_size = imagesets_.size();
  for (int item_id = 0; item_id < batch_size; ++item_id) {
    // get a blob
    timer.Start();
    CHECK_GT(imagesets_size, imagesets_id_);
	for (int image_i = 0; image_i < num_images; ++ image_i) {
		int image_id = (caffe_rng_rand() % imagesets_[imagesets_id_].second.size());
		cv::Mat cv_img = ReadImageToCVMat(root_folder + imagesets_[imagesets_id_].second[image_id],
				new_height, new_width, is_color);
		CHECK(cv_img.data) << "Could not load " << imagesets_[imagesets_id_].second[image_id];
		read_time += timer.MicroSeconds();
		timer.Start();
		// Apply transformations (mirror, crop...) to the image
		int offset = batch->data_.offset(item_id * num_images + image_i);
		this->transformed_data_.set_cpu_data(prefetch_data + offset);
		this->data_transformer_->Transform(cv_img, &(this->transformed_data_));
		trans_time += timer.MicroSeconds();
	}

    prefetch_label[item_id] = imagesets_[imagesets_id_].first;
    // go to the next iter
    imagesets_id_++;
    if (imagesets_id_ >= imagesets_size) {
      // We have reached the end. Restart from the first.
      DLOG(INFO) << "Restarting data prefetching from start.";
      imagesets_id_ = 0;
      if (this->layer_param_.kimage_data_param().shuffle()) {
        ShuffleImages();
      }
    }
  }

  top_shape[0] = batch_size;
  top_shape[1] *= num_images;
  batch->data_.Reshape(top_shape);

  batch_timer.Stop();
  DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
  DLOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
  DLOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";
}

INSTANTIATE_CLASS(KImageDataLayer);
REGISTER_LAYER_CLASS(KImageData);

}  // namespace caffe
#endif  // USE_OPENCV
