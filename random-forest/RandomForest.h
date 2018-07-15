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
	int tree_num;                          // CART ����
	int max_depth;                         // CART ������
	int feature_total;                     // ��������
	int random_sample_num;                 // ÿ�������ȡ��������Ŀ
	int random_feature_num;                // ÿ�������ȡ��������Ŀ
	int node_sample_num_threshold;         // CART �ڵ�������Ŀ����Сֵ

	// public methods
	void train                             // ѵ��
		(const vector<Sample> &train_samples);
	float predict                          // Ԥ��
		(const X &x);

private:
	// train data
	vector<Sample> train_samples;          // ѵ������
	vector<int> sample_indices;            // ѵ���������������������ȡ������
	vector<int> feature_indices;           // �������������������ȡ������
	
	// models
	vector<CART> carts;                    // CART ����

	// private methods
	vector<Sample*> random_select_samples  // �����ȡ random_sample_num ������
		(void);
	vector<int> random_select_features     // �����ȡ random_feature_num ��������0~200��
		(void);
	CART generate_cart                     // ѵ��һ�� CART
		(vector<Sample*> &samples, vector<int> &features);
	void split_node_recursively            // �ݹ�ط��ѽڵ�
		(vector<Sample*> &samples, vector<int> &features, Node *&node, const int depth);
	tuple<float, float> find_split         // �ҵ�һ�������е���ѷָ��
		(vector<Sample*> &samples, const int feature);
	float compute_variance                 // ������������
		(const vector<Sample*> &samples);
	void sort_on_feature                   // ����������ĳ��������������
		(vector<Sample*> &samples, const int feature);
};
