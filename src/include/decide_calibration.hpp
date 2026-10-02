#pragma once

// Per-model calibration for yes/no (binary) probabilities.
//
// Platt scaling: p' = sigmoid(a * logit(p) + b), with a > 0 so the ranking of
// the model is preserved and only the cut-off and sharpness move. A model is
// registered with MAP {'calibration': 'platt:a,b'}; decide_fit_calibration(p, y)
// fits a and b from labelled data and returns that exact spec string.
// Choice and score answers are never touched.

#include "duckdb/common/common.hpp"

#include <cstdint>
#include <string>

namespace duckdb {
namespace anofox {

struct DecidePlatt {
	bool set = false;
	double a = 1.0;
	double b = 0.0;
};

// Parse "platt:<a>,<b>" (a finite and > 0, b finite). On failure returns false
// and sets `why` to a short reason that echoes the offending value.
bool DecideParsePlatt(const string &spec, DecidePlatt &out, string &why);

// "platt:<a>,<b>" with enough digits to round-trip (the string decide_fit_calibration returns).
string DecideFormatPlatt(double a, double b);

// sigmoid(a * logit(clamp(p, 1e-6, 1 - 1e-6)) + b).
double DecidePlattApply(const DecidePlatt &c, double p);

// Maximum-likelihood fit of (a, b) by Newton's method with Platt's smoothed
// targets ((N+ + 1) / (N+ + 2) and 1 / (N- + 2), which keep the fit finite on
// separable data). Fails with a reason when an outcome class is missing, the
// probabilities are all identical, or the fitted slope is not positive (the
// model ranks outcomes no better than chance, which scaling cannot repair).
bool DecideFitPlatt(const vector<double> &p, const vector<uint8_t> &y, double &a, double &b, string &why);

} // namespace anofox
} // namespace duckdb
