#pragma once

#include <vector>
#include <algorithm>
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

	// public methods
	void train(const vector<Sample> &train_samples);
	Y predict(const X &x);

private:
	// data
	vector<Sample> train_samples;
	
	// models
	vector<CART> carts;

	// private methods
	vector<Sample> random_select_samples(void);   // 随机抽取 random_sample_num 个样本
	vector<int> random_select_features(void);  // 随机抽取 random_feature_num 个特征（0~200）
	CART generate_cart(const vector<Sample> &samples, const vector<int> &features);
	void split_node_recursively(const vector<Sample> &samples, const vector<int> &features, Node *node);
	bool need_split_node(Node *node);
};
