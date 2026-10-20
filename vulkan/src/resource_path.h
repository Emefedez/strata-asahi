// vulkan/src/resource_path.h - metal/src/resource_path.h for Linux: the same lookup (environment variable, then
// <exe dir>/metal/<sub>, then the compiled-in path), with the executable found through /proc/self/exe. First on
// the Vulkan build's include path, so the shared sources that include "resource_path.h" get this one.
#pragma once

#include <sys/stat.h>
#include <unistd.h>

#include <climits>
#include <cstdlib>
#include <string>

namespace strata_metal {

inline std::string exe_dir() {
    char raw[PATH_MAX];
    const ssize_t n = readlink("/proc/self/exe", raw, sizeof(raw) - 1);
    if (n <= 0) return "";
    raw[n] = 0;
    std::string s(raw);
    const size_t slash = s.rfind('/');
    return slash == std::string::npos ? "" : s.substr(0, slash);
}

inline bool exists(const std::string& path) {
    struct stat st;
    return stat(path.c_str(), &st) == 0;
}

inline std::string resource(const char* env, const char* sub, const char* compiled) {
    if (env) {
        if (const char* e = std::getenv(env)) return e;
    }
    const std::string dir = exe_dir();
    if (!dir.empty()) {
        const std::string beside = dir + "/metal/" + sub;
        if (exists(beside)) return beside;
    }
    return compiled;
}

}  // namespace strata_metal
