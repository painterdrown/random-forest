#include "stdafx.h"
#include "RandomForest.h"

void RandomForest::train(const vector<X> &train_x, const vector<Y> &train_y) {
	this->train_x = train_x;
	this->train_y = train_y;
}

vector<Y> RandomForest::predict(const vector<X> &test_x) {
	this->test_x = test_x;
}
