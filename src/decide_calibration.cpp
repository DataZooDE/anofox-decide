// decide_calibration.cpp — Platt scaling math (pure; unit-tested in Catch2).

#include "decide_calibration.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>

namespace duckdb {
namespace anofox {

namespace {

constexpr double kEps = 1e-6;

double Logit(double p) {
	p = std::min(std::max(p, kEps), 1.0 - kEps);
	return std::log(p / (1.0 - p));
}

double Sigmoid(double z) {
	if (z >= 0) {
		return 1.0 / (1.0 + std::exp(-z));
	}
	const double e = std::exp(z);
	return e / (1.0 + e);
}

// Parses the whole token as a double.
bool ParseDouble(const string &token, double &out) {
	if (token.empty()) {
		return false;
	}
	char *end = nullptr;
	out = std::strtod(token.c_str(), &end);
	return end && *end == '\0';
}

string Trim(const string &s) {
	size_t b = s.find_first_not_of(" \t\n\r");
	if (b == string::npos) {
		return "";
	}
	size_t e = s.find_last_not_of(" \t\n\r");
	return s.substr(b, e - b + 1);
}

} // namespace

bool DecideParsePlatt(const string &spec_in, DecidePlatt &out, string &why) {
	const string spec = Trim(spec_in);
	static const string prefix = "platt:";
	if (spec.compare(0, prefix.size(), prefix) != 0) {
		why = "'" + spec_in + "' is not a calibration spec; the only supported method is platt";
		return false;
	}
	const string rest = spec.substr(prefix.size());
	const auto comma = rest.find(',');
	double a, b;
	if (comma == string::npos || !ParseDouble(Trim(rest.substr(0, comma)), a) ||
	    !ParseDouble(Trim(rest.substr(comma + 1)), b)) {
		why = "'" + spec_in + "' needs two numbers, platt:<a>,<b>";
		return false;
	}
	if (!std::isfinite(a) || !std::isfinite(b)) {
		why = "'" + spec_in + "' has a non-finite number";
		return false;
	}
	if (a <= 0.0) {
		why = "the slope a must be greater than 0, got " + Trim(rest.substr(0, comma)) +
		      " (a negative slope would invert the model's ranking)";
		return false;
	}
	out.set = true;
	out.a = a;
	out.b = b;
	return true;
}

string DecideFormatPlatt(double a, double b) {
	char buf[96];
	std::snprintf(buf, sizeof(buf), "platt:%.9g,%.9g", a, b);
	return string(buf);
}

double DecidePlattApply(const DecidePlatt &c, double p) {
	if (!c.set) {
		return p;
	}
	return Sigmoid(c.a * Logit(p) + c.b);
}

bool DecideFitPlatt(const vector<double> &p, const vector<uint8_t> &y, double &a, double &b, string &why) {
	const size_t n = p.size();
	size_t pos = 0;
	for (auto v : y) {
		pos += v ? 1 : 0;
	}
	const size_t neg = n - pos;
	if (pos == 0 || neg == 0) {
		why = "all " + std::to_string(n) + " rows have the same outcome (" + (pos ? "true" : "false") +
		      "); calibration needs both";
		return false;
	}
	vector<double> f(n);
	double fmin = 1e300, fmax = -1e300;
	for (size_t i = 0; i < n; i++) {
		f[i] = Logit(p[i]);
		fmin = std::min(fmin, f[i]);
		fmax = std::max(fmax, f[i]);
	}
	if (fmax - fmin < 1e-9) {
		why = "all probabilities are identical (" + std::to_string(p[0]) +
		      "), so there is nothing to calibrate (the stub model always answers 0.5)";
		return false;
	}
	const double hi = (double)(pos + 1) / (double)(pos + 2);
	const double lo = 1.0 / (double)(neg + 2);
	vector<double> t(n);
	for (size_t i = 0; i < n; i++) {
		t[i] = y[i] ? hi : lo;
	}
	auto loss = [&](double A, double B) {
		double s = 0.0;
		for (size_t i = 0; i < n; i++) {
			const double z = A * f[i] + B;
			// -t*log(sigma(z)) - (1-t)*log(1-sigma(z)), computed stably
			s += (z >= 0) ? (1.0 - t[i]) * z + std::log1p(std::exp(-z)) : -t[i] * z + std::log1p(std::exp(z));
		}
		return s;
	};
	double A = 1.0, B = 0.0;
	double fval = loss(A, B);
	for (int iter = 0; iter < 200; iter++) {
		double g1 = 0, g2 = 0, h11 = 1e-12, h12 = 0, h22 = 1e-12;
		for (size_t i = 0; i < n; i++) {
			const double q = Sigmoid(A * f[i] + B);
			const double d = q - t[i];
			const double w = q * (1.0 - q);
			g1 += d * f[i];
			g2 += d;
			h11 += w * f[i] * f[i];
			h12 += w * f[i];
			h22 += w;
		}
		if (std::fabs(g1) < 1e-9 && std::fabs(g2) < 1e-9) {
			break;
		}
		const double det = h11 * h22 - h12 * h12;
		if (!(std::fabs(det) > 1e-300)) {
			break;
		}
		const double dA = -(h22 * g1 - h12 * g2) / det;
		const double dB = -(-h12 * g1 + h11 * g2) / det;
		const double gd = g1 * dA + g2 * dB;
		double step = 1.0;
		bool moved = false;
		while (step >= 1e-12) {
			const double nA = A + step * dA, nB = B + step * dB;
			const double nf = loss(nA, nB);
			if (nf < fval + 1e-4 * step * gd) {
				A = nA;
				B = nB;
				fval = nf;
				moved = true;
				break;
			}
			step /= 2.0;
		}
		if (!moved) {
			break;
		}
	}
	if (!std::isfinite(A) || !std::isfinite(B)) {
		why = "the fit did not converge";
		return false;
	}
	if (A <= 0.0) {
		why = "the fitted slope is not positive (a = " + std::to_string(A) +
		      "): these probabilities rank the outcomes no better than chance, which scaling cannot repair; check "
		      "the labels or use another model";
		return false;
	}
	a = A;
	b = B;
	return true;
}

} // namespace anofox
} // namespace duckdb
