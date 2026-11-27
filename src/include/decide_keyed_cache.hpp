#pragma once

//===----------------------------------------------------------------------===//
// decide_keyed_cache.hpp — a cache of expensive values (local model sessions) that builds each key once, never blocks a lookup or another
// key's build behind a slow build, and keeps one variant per group.
//
// - The cache's own lock is held only to look up, insert and evict: a build (reading a model, loading weights, creating a GPU plugin
//   session: seconds) runs under a lock of its key alone, so a first load of model A does not stall threads scoring model B.
// - Two threads asking for the same missing key build it once: the second waits on the key's lock and then finds the value.
// - A value belongs to a group (one model): inserting a key evicts the group's other keys (the same model built for another thread count,
//   engine or device), so a SET never leaves a second copy of the weights behind. Values already handed out live on until released.
// - The bytes of the cached values are summed, for the object cache's memory accounting.
//===----------------------------------------------------------------------===//

#include <atomic>
#include <cstddef>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace duckdb {
namespace anofox {

//! `P` is the shared pointer type handed out (std::shared_ptr<V>, or duckdb::shared_ptr<V> inside the extension).
template <class V, class P = std::shared_ptr<V>>
class DecideKeyedCache {
public:
	struct Built {
		P value;
		size_t bytes = 0;
	};

	//! The value for `key`, built by `build` if missing (outside the cache lock). `group` names what the key is a variant of.
	P GetOrBuild(const std::string &key, const std::string &group, const std::function<Built()> &build) {
		std::shared_ptr<std::mutex> key_lock;
		{
			std::lock_guard<std::mutex> guard(lock_);
			auto it = slots_.find(key);
			if (it != slots_.end()) {
				return it->second.value;
			}
			auto &building = building_[key];
			if (!building) {
				building = std::make_shared<std::mutex>();
			}
			key_lock = building;
		}
		std::lock_guard<std::mutex> key_guard(*key_lock);
		{
			std::lock_guard<std::mutex> guard(lock_);
			auto it = slots_.find(key);
			if (it != slots_.end()) {
				return it->second.value; // another thread built it while this one waited
			}
		}
		Built built = build(); // may throw: nothing is cached, the next call tries again
		std::lock_guard<std::mutex> guard(lock_);
		for (auto it = slots_.begin(); it != slots_.end();) {
			if (it->second.group == group && it->first != key) {
				bytes_ -= it->second.bytes;
				it = slots_.erase(it);
			} else {
				++it;
			}
		}
		slots_[key] = Slot {built.value, group, built.bytes};
		bytes_ += built.bytes;
		building_.erase(key);
		return built.value;
	}

	//! Drops every value `drop` selects (handed-out values live on).
	void EraseIf(const std::function<bool(const V &)> &drop) {
		std::lock_guard<std::mutex> guard(lock_);
		for (auto it = slots_.begin(); it != slots_.end();) {
			if (drop(*it->second.value)) {
				bytes_ -= it->second.bytes;
				it = slots_.erase(it);
			} else {
				++it;
			}
		}
	}

	//! A snapshot of the cached (key, value) pairs.
	std::vector<std::pair<std::string, P>> Entries() const {
		std::lock_guard<std::mutex> guard(lock_);
		std::vector<std::pair<std::string, P>> out;
		for (const auto &kv : slots_) {
			out.emplace_back(kv.first, kv.second.value);
		}
		return out;
	}

	size_t Bytes() const {
		return bytes_.load();
	}

private:
	struct Slot {
		P value;
		std::string group;
		size_t bytes = 0;
	};
	mutable std::mutex lock_;
	std::map<std::string, Slot> slots_;
	std::map<std::string, std::shared_ptr<std::mutex>> building_;
	std::atomic<size_t> bytes_ {0};
};

} // namespace anofox
} // namespace duckdb
