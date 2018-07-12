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

pair<X, Y> parse_train_line(string line) {
	stringstream ss(line);
	Y y;
	X x;
	x.resize(FEATURE_TOTAL);
	ss >> y;
	while (ss) {
		int index;
		double value;
		char _;
		ss >> index >> _ >> value;
		x[index] = value;
	}
	return pair<X, Y>(x, y);
}

pair<vector<X>, vector<Y>> read_train_data(const char *path) {
	ifstream ifs(path);
	string line;
	vector<X> train_x;
	vector<Y> train_y;
	while (getline(ifs, line)) {
		auto x_y = parse_train_line(line);
		train_x.push_back(x_y.first);
		train_y.push_back(x_y.second);
	}
	ifs.close();
	return pair<vector<X>, vector<Y>>(train_x, train_y);
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

int main() {
	// read train data
	rf_log("begin to read train data");
	auto train_data = read_train_data(TRAIN_DATA_PATH);
	auto train_x = train_data.first;
	auto train_y = train_data.second;

	// train
	rf_log("begin to train");
	RandomForest rf;
	rf.train(train_x, train_y);
	
	// read test data
	rf_log("begin to read test data");
	auto test_x = read_test_data(TEST_DATA_PATH);

	// predict
	rf_log("begin to predict");
	auto test_y = rf.predict(test_x);

	// write predict data to file
	write_predict_data(test_y, PREDICT_DATA_PATH);
	rf_log("predict data is saved at file");

    return 0;
}
