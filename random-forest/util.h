#pragma once

#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>

#define FEATURE_TOTAL 201

using namespace std;

typedef vector<float> X;
typedef bool Y;
typedef struct {
	X x;
	Y y;
} Sample;

string now(void);

void log(const char *log_msg);

int rand(const int begin, const int end);

Sample parse_train_line(const string &line);

vector<Sample> read_train_data(const char *path);

X parse_test_line(const string &line);

vector<X> read_test_data(const char *path);

void write_predict_data(const vector<Y> &test_y, const char *path);
