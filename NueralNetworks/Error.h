#pragma once
#include <stdio.h>
#include <assert.h>
#include <exception>
#define STRFY(X) #X
#define LINE2STR(X) STRFY(X)
#define ERR_STR(MSG) __FILE__ "::" LINE2STR(__LINE__) "::" MSG "\n"
class UndefinedException : public std::exception
{
	const char * msg;
public:
	UndefinedException(const char * details) {
		msg = details;
		fprintf(stderr, msg);
	}
	const char * what() const noexcept override {
		return msg;
	}
};