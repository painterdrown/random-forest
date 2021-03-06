// random-forest.cpp: 定义控制台应用程序的入口点。
//

#include "stdafx.h"
#include "util.h"
#include "RandomForest.h"

#define TRAIN_DATA_PATH    "C:\\Users\\painterdrown\\codespace\\random-forest\\random-forest\\data\\train.txt"
#define TEST_DATA_PATH     "C:\\Users\\painterdrown\\codespace\\random-forest\\random-forest\\data\\test.txt"
#define PREDICT_DATA_PATH  "C:\\Users\\painterdrown\\codespace\\random-forest\\random-forest\\data\\predict.txt"

char main_log[256];

int main(void) {
	// read train data
	sprintf_s(main_log, "begin to read train data from file: %s", TRAIN_DATA_PATH); log(main_log);
	auto train_samples = read_train_data(TRAIN_DATA_PATH);

	// train
	sprintf_s(main_log, "begin to train"); log(main_log);
	RandomForest rf;
	rf.tree_num = 16;
	rf.max_depth = 8;
	rf.feature_total = 201;
	rf.random_sample_num = 65536;
	rf.random_feature_num = 128;
	rf.node_sample_num_threshold = 8;
	rf.train(train_samples);

	// read test data
	sprintf_s(main_log, "begin to read test data from file: %s", TEST_DATA_PATH); log(main_log);
	auto test_x = read_test_data(TEST_DATA_PATH);

	// predict
	sprintf_s(main_log, "begin to predict"); log(main_log);
	vector<float> test_y;
	for (const auto &x : test_x) {
		auto y = rf.predict(x);
		test_y.push_back(y);
	}

	// write predict data to file
	write_predict_data(test_y, PREDICT_DATA_PATH);
	sprintf_s(main_log, "predict data is saved at file: %s", PREDICT_DATA_PATH); log(main_log);

    return 0;
}
