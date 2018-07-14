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

float RandomForest::compute_gini(const vector<Sample*> &samples) {
	int positive_count = 0;
	for (const auto &sample : samples) {
		if (sample->y) ++positive_count;
	}
	float p1 = (float)positive_count / samples.size();
	float p0 = 1.0f - p1;
	float gini = 1.0f - pow(p0, 2) - pow(p1, 2);
	return gini;
}

void RandomForest::split_node_recursively(vector<Sample*> &samples, vector<int> &features, Node *&node, const int depth) {
	if (depth >= max_depth) return;
	if (samples.size() < node_sample_num_threshold) return;
	
	// compute current node gini
	float current_gini = compute_gini(samples);

	// find the feature with smallest gini
	float best_split_point = 0.0f;
	float min_gini = current_gini;
	int min_index = -1;
	for (int i = 0; i < features.size(); ++i) {
		int feature = features[i];
		if (feature == -1) continue;
		auto split_info = find_split(samples, feature);
		float split_point = get<0>(split_info);
		float gini = get<1>(split_info);
		if (gini < min_gini) {
			min_gini = gini;
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
	sprintf_s(rf_log, "split node: feature=%d\tsplit_point=%f\tdepth=%d\tGINI: %f -> %f", features[min_index], best_split_point, depth+1, current_gini, min_gini); log(rf_log);
	
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
		int positive_count = 0;
		int negative_count = 0;
		for (const auto &sample : samples) {
			if (sample->y) ++positive_count;
			else ++negative_count;
		}
		node->value = positive_count > negative_count;
	}
}

tuple<float, float> RandomForest::find_split(vector<Sample*> &samples, const int feature) {
	sort_on_feature(samples, feature);
	int positive_total = 0;
	for (const auto &sample : samples) positive_total += sample->y ? 1 : 0;
	int positive_count1 = 0;
	int positive_count2 = positive_total - positive_count1;
	float min_gini = 1.0f;
	int min_index = -1;
	float best_split_point = 0.0f;
	for (int i = 0; i < samples.size() - 1; ++i) {
		float split_point = (samples[i]->x[feature] + samples[i + 1]->x[feature]) / 2;
		int total1 = i + 1;
		int total2 = samples.size() - total1;
		positive_count1 += samples[i]->y ? 1 : 0;
		positive_count2 = positive_total - positive_count1;
		float gini1 = 1 - pow((float)positive_count1 / total1, 2) - pow(float(total1 - positive_count1) / total1, 2);
		float gini2 = 1 - pow((float)positive_count2 / total2, 2) - pow(float(total2 - positive_count2) / total2, 2);
		float gini = (float)total1 / samples.size() * gini1 + (float)total2 / samples.size() * gini2;
		if (gini < min_gini) {
			min_gini = gini;
			min_index = i;
			best_split_point = split_point;
		}
	}
	return make_tuple(best_split_point, min_gini);
}

CART RandomForest::generate_cart(vector<Sample*> &samples, vector<int> &features) {
	CART cart;
	split_node_recursively(samples, features, cart.root, 0);
	return cart;
}

void RandomForest::sort_on_feature(vector<Sample*> &samples, const int feature) {
	sort(samples.begin(), samples.end(), [=](const auto &s1, const auto &s2) { return s1->x[feature] < s2->x[feature]; });
}
