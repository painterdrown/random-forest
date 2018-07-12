#pragma once

#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <string>

using namespace std;

typedef vector<float> X;
typedef bool Y;
typedef struct {
	X x;
	Y y;
} Sample;

string now(void);

void rf_log(string msg);

int rand(const int begin, const int end);
