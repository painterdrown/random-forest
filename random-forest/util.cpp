#include "stdafx.h"
#include "util.h"

char util_log[256];

string now(void) {
	struct tm t;
	time_t n;
	time(&n);
	localtime_s(&t, &n);
	char result[32];
	strftime(result, sizeof(result), "%Y-%m-%d %H:%M:%S", &t);
	return result;
}

void log(const char *msg) {
	printf("[%s] %s\n", now().c_str(), msg);
}

int rand(const int begin, const int end) {
	srand(time(NULL));

	return rand() % (end - begin) + begin;
}

Sample parse_train_line(const string &line) {
	stringstream ss(line);
	Sample sample;
	sample.x.resize(FEATURE_TOTAL);
	ss >> sample.y;
	while (ss) {
		int index;
		float value;
		char _;
		ss >> index >> _ >> value;
		sample.x[index-1] = value;
	}
	return sample;
}

vector<Sample> read_train_data(const char *path) {
	ifstream ifs(path);
	string line;
	vector<Sample> train_samples;
	int count = 0;
	while (getline(ifs, line)) {
		auto sample = parse_train_line(line);
		train_samples.push_back(sample);
		count++;
		if (count % 10000 == 0) {
			sprintf_s(util_log, "%d train samples are read", count); log(util_log);
		}
	}
	sprintf_s(util_log, "%d train samples are read totally", count); log(util_log);
	ifs.close();
	return train_samples;
}

X parse_test_line(const string &line) {
	stringstream ss(line);
	int _;
	X x;
	x.resize(FEATURE_TOTAL);
	ss >> _;
	while (ss) {
		int index;
		float value;
		char _;
		ss >> index >> _ >> value;
		x[index-1] = value;
	}
	return x;
}

vector<X> read_test_data(const char *path) {
	ifstream ifs(path);
	string line;
	vector<X> test_x;
	int count = 0;
	while (getline(ifs, line)) {
		auto x = parse_test_line(line);
		test_x.push_back(x);
		count++;
		if (count % 10000 == 0) {
			sprintf_s(util_log, "%d test samples are read", count); log(util_log);
		}
	}
	sprintf_s(util_log, "%d test samples are read totally", count); log(util_log);
	ifs.close();
	return test_x;
}

void write_predict_data(const vector<float> &test_y, const char *path) {
	ofstream ofs(path);
	ofs << "id,label\n";
	for (int i = 0; i < test_y.size(); ++i) {
		ofs << i << ',' << test_y[i] << '\n';
	}
	ofs.close();
}

void shuffle(vector<int> &cards) {
	srand(time(NULL));

	int n = cards.size();
	for (int i = 0; i < n; ++i) {
		int index = rand() % (n - i) + i;
		if (index != i) {
			int tmp = cards[i];
			cards[i] = cards[index];
			cards[index] = tmp;
		}

	}

}