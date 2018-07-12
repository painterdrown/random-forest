#pragma once

#include <vector>

using namespace std;

typedef vector<float> X;
typedef bool Y;

class RandomForest {

public:
	void train(const vector<X> &train_x, const vector<Y> &train_y);
	vector<Y> predict(const vector<X> &test_x);

private:
	vector<X> train_x;
	vector<Y> train_y;
	vector<X> test_x;

};
