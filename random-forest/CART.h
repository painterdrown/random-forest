#pragma once

#include <vector>
#include "util.h"

using namespace std;

struct Node {
	int feature;
	Y value;
	float split_point;
	Node *left;
	Node *right;

	Node() {
		feature = -1;
		value = false;
		split_point = 0.0f;
		left = NULL;
		right = NULL;
	}

	bool is_leaf() const {
		return left == NULL && right == NULL;
	}
};

struct CART {
	Node *root;

	// constructor
	CART() {
		root = NULL;
	}

	// classify
	Y classify(const X &x) const {
		Node *node = root;
		while (!node->is_leaf()) {  // while not a leaf node
			if (node->left == NULL) node = node->right;
			else if (node->right == NULL) node = node->left;
			else {
				node = x[node->feature] <= node->split_point ? node->left : node->right;
			}
		}
		return node->value;
	}
};
