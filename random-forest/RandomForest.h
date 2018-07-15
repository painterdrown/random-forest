#pragma once

#include <vector>
#include <algorithm>
#include <tuple>
#include "CART.h"
#include "util.h"

using namespace std;

class RandomForest {

public:
	// params
	int tree_num;                          // CART 总数
	int max_depth;                         // CART 最大深度
	int feature_total;                     // 特征总数
	int random_sample_num;                 // 每次随机抽取样本的数目
	int random_feature_num;                // 每次随机抽取特征的数目
	int node_sample_num_threshold;         // CART 节点样本数目的最小值

	// public methods
	void train                             // 训练
		(const vector<Sample> &train_samples);
	float predict                          // 预测
		(const X &x);

private:
	// train data
	vector<Sample> train_samples;          // 训练样本
	vector<int> sample_indices;            // 训练样本索引（用于随机抽取样本）
	vector<int> feature_indices;           // 特征索引（用于随机抽取特征）
	
	// models
	vector<CART> carts;                    // CART 集合

	// private methods
	vector<Sample*> random_select_samples  // 随机抽取 random_sample_num 个样本
		(void);
	vector<int> random_select_features     // 随机抽取 random_feature_num 个特征（0~200）
		(void);
	CART generate_cart                     // 训练一颗 CART
		(vector<Sample*> &samples, vector<int> &features);
	void split_node_recursively            // 递归地分裂节点
		(vector<Sample*> &samples, vector<int> &features, Node *&node, const int depth);
	tuple<float, float> find_split         // 找到一组特征中的最佳分割点
		(vector<Sample*> &samples, const int feature);
	float compute_variance                 // 计算样本方差
		(const vector<Sample*> &samples);
	void sort_on_feature                   // 将样本基于某个特征进行排序
		(vector<Sample*> &samples, const int feature);
};
