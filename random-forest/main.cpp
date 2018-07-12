// random-forest.cpp: 定义控制台应用程序的入口点。
//

#include "stdafx.h"
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include "RandomForest.h"
#include "util.h"

#define TRAIN_DATA_PATH    "C:\\Users\\painterdrown\\codespace\\random-forest\\random-forest\\data\\train.txt"
#define TEST_DATA_PATH     "C:\\Users\\painterdrown\\codespace\\random-forest\\random-forest\\data\\test.txt"
#define PREDICT_DATA_PATH  "C:\\Users\\painterdrown\\codespace\\random-forest\\random-forest\\data\\predict.txt"
#define FEATURE_TOTAL  201

using namespace std;

Sample parse_train_line(string line) {
	stringstream ss(line);
	Sample sample;
	sample.x.resize(FEATURE_TOTAL);
	ss >> sample.y;
	while (ss) {
		int index;
		double value;
		char _;
		ss >> index >> _ >> value;
		sample.x[index] = value;
	}
	return sample;
}

vector<Sample> read_train_data(const char *path) {
	ifstream ifs(path);
	string line;
	vector<Sample> train_samples;
	while (getline(ifs, line)) {
		auto sample = parse_train_line(line);
		train_samples.push_back(sample);
	}
	ifs.close();
	return train_samples;
}

X parse_test_line(string line) {
	stringstream ss(line);
	int _;
	X x;
	x.resize(FEATURE_TOTAL);
	ss >> _;
	while (ss) {
		int index;
		double value;
		char _;
		ss >> index >> _ >> value;
		x[index] = value;
	}
	return x;
}

vector<X> read_test_data(const char *path) {
	ifstream ifs(path);
	string line;
	vector<X> test_x;
	while (getline(ifs, line)) {
		auto x = parse_test_line(line);
		test_x.push_back(x);
	}
	ifs.close();
	return test_x;
}

void write_predict_data(vector<Y> test_y, const char *path) {
	ofstream ofs(path);
	ofs << "id,label\n";
	for (int i = 0; i < test_y.size(); ++i) {
		ofs << i << ',' << test_y[i] << '\n';
	}
	ofs.close();
}

int main(void) {
	// read train data
	rf_log("begin to read train data");
	auto train_samples = read_train_data(TRAIN_DATA_PATH);

	//// train
	//rf_log("begin to train");
	//RandomForest rf;
	//rf.tree_num = 32;
	//rf.max_depth = 6;
	//rf.feature_total = FEATURE_TOTAL;
	//rf.random_sample_num = 10000;
	//rf.random_feature_num = 16;
	//rf.train(train_samples);

	// read test data
	rf_log("begin to read test data");
	auto test_x = read_test_data(TEST_DATA_PATH);

	//// predict
	//rf_log("begin to predict");
	//vector<Y> test_y;
	//for (const auto &x : test_x) {
	//	auto y = rf.predict(x);
	//	test_y.push_back(y);
	//}

	//// write predict data to file
	//write_predict_data(test_y, PREDICT_DATA_PATH);
	//rf_log("predict data is saved at file");

    return 0;
}
