#pragma once

#include <vector>
#include <algorithm>
#include <tuple>
#include "CART.h"
#include "util.h"

#define INFO_GAIN_THRESHOLD   0.1
#define SAMPLE_NUM_THRESHOLD  10

using namespace std;

class RandomForest {

public:
	// params
	int tree_num;
	int max_depth;
	int feature_total;
	int random_sample_num;
	int random_feature_num;
	int node_sample_num_threshold;

	// public methods
	void train(const vector<Sample> &train_samples);
	float predict(const X &x);

private:
	// data
	vector<Sample> train_samples;
	vector<int> sample_indices;
	vector<int> feature_indices;
	
	// models
	vector<CART> carts;

	// private methods
	vector<Sample*> random_select_samples(void);   // 随机抽取 random_sample_num 个样本
	vector<int> random_select_features(void);      // 随机抽取 random_feature_num 个特征（0~200）
	CART generate_cart(vector<Sample*> &samples, vector<int> &features);
	void split_node_recursively(vector<Sample*> &samples, vector<int> &features, Node *&node, const int depth);
	tuple<float, float> find_split(vector<Sample*> &samples, const int feature);
	float compute_variance(const vector<Sample*> &samples);
	void sort_on_feature(vector<Sample*> &samples, const int feature);
};
