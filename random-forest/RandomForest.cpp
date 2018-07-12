#include "stdafx.h"
#include "RandomForest.h"

void RandomForest::train(const vector<Sample> &train_samples) {
	this->train_samples = train_samples;

	carts.resize(tree_num);
	for (int i = 0; i < tree_num; ++i) {
		auto samples = random_select_samples();
		auto features = random_select_features();
		auto cart = generate_cart(samples, features);
		carts.push_back(cart);
	}
}

Y RandomForest::predict(const X &x) {
	int positive_votes = 0;
	int negative_votes = 0;

	for (const auto &cart : carts) {
		Y vote = cart.classify(x);
		if (vote) ++positive_votes;
		else ++negative_votes;
	}

	return positive_votes > negative_votes;
}

vector<Sample> RandomForest::random_select_samples(void) {
	vector<Sample> samples;
	vector<int> samples_index;
	const int train_samples_total = train_samples.size();
	for (int i = 0; i < random_sample_num; ++i) {
		int r = rand(0, train_samples_total);
		while (find(samples_index.begin(), samples_index.end(), r) != samples_index.end()) {
			r = rand(0, train_samples_total);
		}
		samples.push_back(train_samples[r]);
		samples_index.push_back(r);
	}
	return samples;
}

vector<int> RandomForest::random_select_features(void) {
	vector<int> features;
	for (int i = 0; i < random_feature_num; ++i) {
		int r = rand(0, feature_total);
		while (find(features.begin(), features.end(), r) != features.end()) {
			r = rand(0, feature_total);
		}
		features.push_back(r);
	}
	return features;
}

bool RandomForest::need_split_node(Node *node) {
	return (node->info_gain > INFO_GAIN_THRESHOLD) &&
		(node->sample_num > SAMPLE_NUM_THRESHOLD) &&
		(node->level < max_depth);
}

void RandomForest::split_node_recursively(const vector<Sample> &samples, const vector<int> &features, Node *node) {
	// TODO

	if (need_split_node(node->left)) split_node_recursively(samples, features, node->left);
	if (need_split_node(node->right)) split_node_recursively(samples, features, node->right);
}

CART RandomForest::generate_cart(const vector<Sample> &samples, const vector<int> &features) {
	CART cart;

	// sort on every fearture
	for (int i = 0; i < features.size(); ++i) {

	}

	split_node_recursively(samples, features, cart.root);
}
