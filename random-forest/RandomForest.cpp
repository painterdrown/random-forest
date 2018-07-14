#include "stdafx.h"
#include "RandomForest.h"

char rf_log[256];

void RandomForest::train(const vector<Sample> &train_samples) {
	this->train_samples = train_samples;

	carts.resize(tree_num);
	for (int i = 0; i < tree_num; ++i) {
		sprintf_s(rf_log, "begin to train CART #%d", i + 1); log(rf_log);
		auto samples = random_select_samples();
		auto features = random_select_features();
		carts[i] = generate_cart(samples, features);
	}
}

float RandomForest::predict(const X &x) {
	float p = 0;

	for (const auto &cart : carts) {
		p += cart.classify(x);
	}

	return p / tree_num;
}

vector<Sample*> RandomForest::random_select_samples(void) {
	vector<Sample*> samples;
	vector<int> samples_index;
	const int train_samples_total = train_samples.size();
	for (int i = 0; i < random_sample_num; ++i) {
		int r = rand(0, train_samples_total);
		while (find(samples_index.begin(), samples_index.end(), r) != samples_index.end()) {
			r = rand(0, train_samples_total);
		}
		samples.push_back(&train_samples[r]);
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

float RandomForest::compute_variance(const vector<Sample*> &samples) {
	int t = samples.size();
	int p = 0;
	for (const auto &sample : samples) if (sample->y) ++p;
	float mean = (float)p / t;
	float variance = (float)p / t * pow(1 - mean, 2) + (float)(t - p) / t * pow(0 - mean, 2);
	return variance;
}

void RandomForest::split_node_recursively(vector<Sample*> &samples, vector<int> &features, Node *&node, const int depth) {
	if (depth >= max_depth) return;
	if (samples.size() < node_sample_num_threshold) return;
	
	// compute current node gini
	float current_variance = compute_variance(samples);

	// find the feature with smallest variance
	float best_split_point = 0.0f;
	float min_variance = current_variance;
	int min_index = -1;
	for (int i = 0; i < features.size(); ++i) {
		int feature = features[i];
		if (feature == -1) continue;
		auto split_info = find_split(samples, feature);
		float split_point = get<0>(split_info);
		float variance = get<1>(split_info);
		if (variance < min_variance) {
			min_variance = variance;
			min_index = i;
			best_split_point = split_point;
		}
	}

	if (min_index == -1) return;

	// split
	node = new Node();
	Node *test = carts[0].root;
	node->feature = features[min_index];
	node->split_point = best_split_point;
	sprintf_s(rf_log, "split node: feature=%d\tsplit_point=%f\tdepth=%d\tvariance: %f -> %f", features[min_index], best_split_point, depth+1, current_variance, min_variance); log(rf_log);
	
	// split left and right nodes
	vector<Sample*> l_samples, r_samples;
	for (auto s : samples) {
		if (s->x[features[min_index]] <= best_split_point) l_samples.push_back(s);
		else r_samples.push_back(s);
	}
	features[min_index] = -1;
	split_node_recursively(l_samples, features, node->left, depth + 1);
	split_node_recursively(r_samples, features, node->right, depth + 1);
	
	// leaf node
	if (node->is_leaf()) {
		int p = 0;
		for (const auto &sample : samples) if (sample->y) ++p;
		node->value =  (float)p / samples.size();
	}
}

tuple<float, float> RandomForest::find_split(vector<Sample*> &samples, const int feature) {
	sort_on_feature(samples, feature);
	int t = samples.size();
	int p = 0;
	for (const auto &sample : samples) p += sample->y ? 1 : 0;
	int p1 = 0;
	int p2 = p - p1;
	float min_variance = 1.0f;
	int min_index = -1;
	float best_split_point = 0.0f;
	for (int i = 0; i < samples.size() - 1; ++i) {
		float split_point = (samples[i]->x[feature] + samples[i + 1]->x[feature]) / 2;
		int t1 = i + 1;
		int t2 = t - t1;
		p1 += samples[i]->y ? 1 : 0;
		p2 = p - p1;
		float mean1 = (float)p1 / t1;
		float mean2 = (float)p2 / t2;
		float variance1 = (float)p1 / t1 * pow(1 - mean1, 2) + (float)(t1 - p1) / t1 * pow(0 - mean1, 2);
		float variance2 = (float)p2 / t2 * pow(1 - mean2, 2) + (float)(t2 - p2) / t2 * pow(0 - mean2, 2);
		float variance = (float)t1 / t * variance1 + (float)t2 / t * variance2;
		if (variance < min_variance) {
			min_variance = variance;
			min_index = i;
			best_split_point = split_point;
		}
	}
	return make_tuple(best_split_point, min_variance);
}

CART RandomForest::generate_cart(vector<Sample*> &samples, vector<int> &features) {
	CART cart;
	split_node_recursively(samples, features, cart.root, 0);
	return cart;
}

void RandomForest::sort_on_feature(vector<Sample*> &samples, const int feature) {
	sort(samples.begin(), samples.end(), [=](const auto &s1, const auto &s2) { return s1->x[feature] < s2->x[feature]; });
}
