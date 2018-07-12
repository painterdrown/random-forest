#include "stdafx.h"
#include "util.h"

string now(void) {
	struct tm t;
	time_t n;
	time(&n);
	localtime_s(&t, &n);
	char result[32];
	strftime(result, sizeof(result), "%Y-%m-%d %H:%M:%S", &t);
	return result;
}

void rf_log(string msg) {
	printf("[%s] %s\n", now().c_str(), msg.c_str());
}
