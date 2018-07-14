// random-forest.cpp: 定义控制台应用程序的入口点。
//

#include "stdafx.h"
#include "util.h"
#include "RandomForest.h"

#define TRAIN_DATA_PATH    "C:\\Users\\painterdrown\\codespace\\random-forest\\random-forest\\data\\train.txt"
#define TEST_DATA_PATH     "C:\\Users\\painterdrown\\codespace\\random-forest\\random-forest\\data\\test.txt"
#define PREDICT_DATA_PATH  "C:\\Users\\painterdrown\\codespace\\random-forest\\random-forest\\data\\predict.txt"

int main(void) {
	// read train data
	rf_log("begin to read train data");
	auto train_samples = read_train_data(TRAIN_DATA_PATH);

	// train
	rf_log("begin to train");
	RandomForest rf;
	rf.tree_num = 32;
	rf.max_depth = 16;
	rf.feature_total = 201;
	rf.random_sample_num = 32768;  // 2^15
	rf.random_feature_num = 16;
	rf.node_sample_num_threshold = 8;
	rf.train(train_samples);

	// read test data
	rf_log("begin to read test data");
	auto test_x = read_test_data(TEST_DATA_PATH);

	// predict
	rf_log("begin to predict");
	vector<Y> test_y;
	for (const auto &x : test_x) {
		auto y = rf.predict(x);
		test_y.push_back(y);
	}

	// write predict data to file
	write_predict_data(test_y, PREDICT_DATA_PATH);
	rf_log("predict data is saved at file");

    return 0;
}
