// Copyright (C) 2020-2025 CERN
// License Apache2 - see LICENCE file

#include "nvmlmon.h"
#include "utils.h"
#include <dlfcn.h>
#include <string>

#define MONITOR_NAME "nvmlmon"

// NVML return code constants
#define NVML_SUCCESS              0
#define NVML_ERROR_NOT_FOUND      6
#define NVML_ERROR_INSUFFICIENT_SIZE 7

// NVML device name buffer size
#define NVML_DEVICE_NAME_BUFFER_SIZE 64

// NVML types
typedef struct nvmlDevice_st* nvmlDevice_t;

typedef int nvmlReturn_t;

typedef struct {
  unsigned long long total;  //!< Total physical device memory (in bytes)
  unsigned long long free;   //!< Unallocated device memory (in bytes)
  unsigned long long used;   //!< Allocated device memory (in bytes)
} nvmlMemory_t;

typedef enum {
  NVML_CLOCK_GRAPHICS = 0,
  NVML_CLOCK_SM = 1,
  NVML_CLOCK_MEM = 2,
  NVML_CLOCK_VIDEO = 3,
} nvmlClockType_t;

// NVML function pointers
// Lifecycle
static nvmlReturn_t (*nvmlInit)(void);
static nvmlReturn_t (*nvmlShutdown)(void);
static const char*  (*nvmlErrorString)(nvmlReturn_t);

// Device
static nvmlReturn_t (*nvmlDeviceGetCount)(unsigned int* deviceCount);
static nvmlReturn_t (*nvmlDeviceGetHandleByIndex)(unsigned int index, nvmlDevice_t* device);

// Process
static nvmlReturn_t (*nvmlDeviceGetProcessUtilization)(nvmlDevice_t device,
                                                       nvmlProcessUtilizationSample_t* utilization,
                                                       unsigned int* processSamplesCount,
                                                       unsigned long long lastSeenTimeStamp);

// Process — highest available version (v3 > v2 > v1) is resolved at load time
static nvmlReturn_t (*nvmlDeviceGetComputeRunningProcesses)(nvmlDevice_t device,
                                                            unsigned int* infoCount,
                                                            nvmlProcessInfo_t* infos);

// Device info
static nvmlReturn_t (*nvmlDeviceGetName)(nvmlDevice_t device, char* name, unsigned int length);
static nvmlReturn_t (*nvmlDeviceGetMaxClockInfo)(nvmlDevice_t device, nvmlClockType_t type,
                                                 unsigned int* clock);
static nvmlReturn_t (*nvmlDeviceGetMemoryInfo)(nvmlDevice_t device, nvmlMemory_t* memory);

// Static member definition
void* nvmlmon::nvml_handle = nullptr;

// Constructor; uses RAII pattern to be valid after construction
nvmlmon::nvmlmon() : nvidia_stats{} {
  log_init(MONITOR_NAME);
#undef MONITOR_NAME
  for (const auto& param : params) {
    nvidia_stats.emplace(param.get_name(), prmon::monitored_value(param));
  }

  // Attempt to load NVML dynamically
  valid = load_nvml_lib();
  if (valid) {
    // Attempt to initialize NVML; if this fails we cannot get data
    valid = init_nvml();
    last_seen_timestamp = 0;

    if (valid) {
      utilization.resize(max_samples);
      memory_info.resize(max_samples);
    }
  }
}

nvmlmon::~nvmlmon() {
  if (valid) {
    nvmlReturn_t result = nvmlShutdown();
    if (result != NVML_SUCCESS) {
      warning("nvmlShutdown was not successfully finished");
    }
  }
}

void nvmlmon::update_stats(const std::vector<pid_t>& pids, const std::string read_path) {
  prmon::monitored_value_map nvidia_stats_update{};
  for (const auto& value : nvidia_stats) {
    nvidia_stats_update[value.first] = 0L;
  }

  nvmlReturn_t result;
  unsigned long long current_max_timestamp = last_seen_timestamp;
  unsigned int active_gpus{0};

  for (unsigned int gpu_idx{0}; gpu_idx < ngpus; ++gpu_idx) {
    nvmlDevice_t device;
    result = nvmlDeviceGetHandleByIndex(gpu_idx, &device);
    if (result != NVML_SUCCESS) {
      warning("Failed to get handle for GPU index " + std::to_string(gpu_idx));
      continue;
    }

    unsigned int util_count = max_samples;
    result = nvmlDeviceGetProcessUtilization(device, utilization.data(), &util_count,
                                             last_seen_timestamp);

    if (result == NVML_ERROR_INSUFFICIENT_SIZE) {
      warning("Utilization sample buffer size (" + std::to_string(max_samples) +
              ") exceeded. Consider increasing max_samples.");
      continue;
    }

    if (result == NVML_ERROR_NOT_FOUND) {
      util_count = 0;
    } else if (result != NVML_SUCCESS) {
      warning("Failed to get process utilization for GPU index " + std::to_string(gpu_idx) +
              ": " + std::string(nvmlErrorString(result)));
      continue;
    }

    for (unsigned int i{0}; i < util_count; ++i) {
      if (utilization[i].timeStamp > current_max_timestamp) {
        current_max_timestamp = utilization[i].timeStamp;
      }
    }

    unsigned int mem_count = max_samples;
    result = nvmlDeviceGetComputeRunningProcesses(device, &mem_count, memory_info.data());

    if (result == NVML_ERROR_INSUFFICIENT_SIZE) {
      warning("Memory sample buffer size (" + std::to_string(max_samples) +
              ") exceeded. Consider increasing max_samples.");
      continue;
    }

    if (result == NVML_ERROR_NOT_FOUND) {
      mem_count = 0;
    } else if (result != NVML_SUCCESS) {
      warning("Failed to get process memory utilization for GPU index " +
              std::to_string(gpu_idx) + ": " + std::string(nvmlErrorString(result)));
      continue;
    }

    bool gpu_is_active{false};
    for (unsigned int target_pid : pids) {
      for (unsigned int i{0}; i < util_count; ++i) {
        if (utilization[i].pid == target_pid) {
          nvidia_stats_update["gpusmpct"] += utilization[i].smUtil;
          nvidia_stats_update["gpumempct"] += utilization[i].memUtil;
          gpu_is_active = true;
          break;
        }
      }
      for (unsigned int i{0}; i < mem_count; ++i) {
        if (memory_info[i].pid == target_pid) {
          nvidia_stats_update["gpufbmem"] += (memory_info[i].usedGpuMemory / B_to_KB);
          gpu_is_active = true;
          break;
        }
      }
    }

    if (gpu_is_active) {
      active_gpus++;
    }
  }

  last_seen_timestamp = current_max_timestamp;
  nvidia_stats_update["ngpus"] = active_gpus;

  for (auto& value : nvidia_stats) {
    if (nvidia_stats_update.count(value.first)) {
      value.second.set_value(nvidia_stats_update[value.first]);
    }
  }
}

prmon::monitored_value_map const nvmlmon::get_text_stats() {
  prmon::monitored_value_map nvidia_stat_map{};
  for (const auto& value : nvidia_stats) {
    nvidia_stat_map[value.first] = value.second.get_value();
  }
  return nvidia_stat_map;
}

prmon::monitored_value_map const nvmlmon::get_json_total_stats() {
  prmon::monitored_value_map nvidia_stat_map{};
  for (const auto& value : nvidia_stats) {
    nvidia_stat_map[value.first] = value.second.get_max_value();
  }
  return nvidia_stat_map;
}

prmon::monitored_average_map const nvmlmon::get_json_average_stats(
    unsigned long long elapsed_clock_ticks) {
  prmon::monitored_average_map nvidia_stat_map{};
  for (const auto& value : nvidia_stats) {
    nvidia_stat_map[value.first] = value.second.get_average_value();
  }
  return nvidia_stat_map;
}

// Open libnvidia-ml.so and resolve all function pointers
bool nvmlmon::load_nvml_lib() {
  nvml_handle = dlopen("libnvidia-ml.so", RTLD_NOW);
  if (!nvml_handle) {
    nvml_handle = dlopen("libnvidia-ml.so.1", RTLD_NOW);
  }
  if (!nvml_handle) {
    return false;
  }

#define LOAD_SYM(var, sym, cast)                      \
  var = (cast)dlsym(nvml_handle, sym);                \
  if (!(var)) { goto load_error; }

  LOAD_SYM(nvmlInit,       "nvmlInit",       nvmlReturn_t (*)())
  LOAD_SYM(nvmlShutdown,   "nvmlShutdown",   nvmlReturn_t (*)())
  LOAD_SYM(nvmlErrorString,"nvmlErrorString",const char* (*)(nvmlReturn_t))
  LOAD_SYM(nvmlDeviceGetCount,
           "nvmlDeviceGetCount",
           nvmlReturn_t (*)(unsigned int*))
  LOAD_SYM(nvmlDeviceGetHandleByIndex,
           "nvmlDeviceGetHandleByIndex",
           nvmlReturn_t (*)(unsigned int, nvmlDevice_t*))
  LOAD_SYM(nvmlDeviceGetProcessUtilization,
           "nvmlDeviceGetProcessUtilization",
           nvmlReturn_t (*)(nvmlDevice_t, nvmlProcessUtilizationSample_t*,
                            unsigned int*, unsigned long long))
  LOAD_SYM(nvmlDeviceGetName,
           "nvmlDeviceGetName",
           nvmlReturn_t (*)(nvmlDevice_t, char*, unsigned int))
  LOAD_SYM(nvmlDeviceGetMaxClockInfo,
           "nvmlDeviceGetMaxClockInfo",
           nvmlReturn_t (*)(nvmlDevice_t, nvmlClockType_t, unsigned int*))
  LOAD_SYM(nvmlDeviceGetMemoryInfo,
           "nvmlDeviceGetMemoryInfo",
           nvmlReturn_t (*)(nvmlDevice_t, nvmlMemory_t*))

#undef LOAD_SYM

  // Try v3 first then v2, then v1
  nvmlDeviceGetComputeRunningProcesses =
      (nvmlReturn_t (*)(nvmlDevice_t, unsigned int*, nvmlProcessInfo_t*))
      dlsym(nvml_handle, "nvmlDeviceGetComputeRunningProcesses_v3");
  if (!nvmlDeviceGetComputeRunningProcesses){
    nvmlDeviceGetComputeRunningProcesses =
        (nvmlReturn_t (*)(nvmlDevice_t, unsigned int*, nvmlProcessInfo_t*))
        dlsym(nvml_handle, "nvmlDeviceGetComputeRunningProcesses_v2");
  }
  if (!nvmlDeviceGetComputeRunningProcesses) {
    nvmlDeviceGetComputeRunningProcesses =
        (nvmlReturn_t (*)(nvmlDevice_t, unsigned int*, nvmlProcessInfo_t*))
        dlsym(nvml_handle, "nvmlDeviceGetComputeRunningProcesses");
  }
  if (!nvmlDeviceGetComputeRunningProcesses) {
    goto load_error;
  }

  return true;
load_error:
  dlclose(nvml_handle);
  nvml_handle = nullptr;
  return false;
}

bool nvmlmon::init_nvml() {
  nvmlReturn_t result = nvmlInit();
  if (result != NVML_SUCCESS) {
    warning("NVML Init failed: " + std::string(nvmlErrorString(result)));
    return false;
  }

  unsigned int gpus{};
  result = nvmlDeviceGetCount(&gpus);
  if (result != NVML_SUCCESS) {
    warning("Failed to get GPU count: " + std::string(nvmlErrorString(result)));
    nvmlShutdown();
    return false;
  }

  if (gpus == 0) {
    warning("nvmlInit() succeeded but no GPUs found");
    nvmlShutdown();
    return false;
  }
  ngpus = gpus;
  return true;
}

prmon::parameter_list const nvmlmon::get_parameter_list() { return params; }

void const nvmlmon::get_hardware_info(nlohmann::json& hw_json) {
  hw_json["HW"]["gpu"]["nGPU"] = ngpus;
  for (unsigned int i{0}; i < ngpus; ++i) {
    nvmlDevice_t device;
    nvmlReturn_t result;
    nvmlMemory_t memInfo;
    char name[NVML_DEVICE_NAME_BUFFER_SIZE] = {};
    unsigned int sm_freq{0};
    unsigned long long total_mem{0};

    result = nvmlDeviceGetHandleByIndex(i, &device);
    if (result != NVML_SUCCESS) {
      warning("Failed to get handle for GPU index " + std::to_string(i));
      continue;
    }

    result = nvmlDeviceGetName(device, name, NVML_DEVICE_NAME_BUFFER_SIZE);
    if (result != NVML_SUCCESS) {
      warning("Failed to get name for GPU index " + std::to_string(i));
      name[0] = '\0';
    }

    result = nvmlDeviceGetMaxClockInfo(device, NVML_CLOCK_SM, &sm_freq);
    if (result != NVML_SUCCESS) {
      warning("Failed to get SM frequency for GPU index " + std::to_string(i));
      sm_freq = 0;
    }

    result = nvmlDeviceGetMemoryInfo(device, &memInfo);
    if (result != NVML_SUCCESS) {
      warning("Failed to get memory info for GPU index " + std::to_string(i));
      memInfo.total = 0;
    }

    total_mem = memInfo.total / B_to_KB;

    std::string gpu_number = "gpu_" + std::to_string(i);
    hw_json["HW"]["gpu"][gpu_number]["name"] = name;
    hw_json["HW"]["gpu"][gpu_number]["sm_freq"] = sm_freq;
    hw_json["HW"]["gpu"][gpu_number]["total_mem"] = total_mem;
  }
}

void const nvmlmon::get_unit_info(nlohmann::json& unit_json) {
  prmon::fill_units(unit_json, params);
}
