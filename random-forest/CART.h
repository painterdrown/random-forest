#pragma once

#include <vector>
#include "util.h"

using namespace std;

struct Node {
	int feature;
	Y value;
	float split_point;
	float info_gain;
	int sample_num;
	int level;
	Node *left;
	Node *right;

	Node() {
		feature = -1;
		value = false;
		split_point = 0.0f;
		info_gain = 0.0f;
		sample_num = 0;
		level = 0;
		left = NULL;
		right = NULL;
	}
};

struct CART {
	Node *root;

	Y classify(const X &x) const {
		Node *node = root;
		while (!node->left || !node->right) {  // while not a leaf node
			node = x[node->feature] <= node->value ? node->left : node->right;
		}
		return node->value;
	}
};
