// Copyright (C) 2020-2025 CERN
// License Apache2 - see LICENCE file
#include <string>

#if defined(ENABLE_NVIDIA_GPU)
#include "nvidiamon.h"
#include <nvml.h>

#include "utils.h"


#define MONITOR_NAME "nvidiamon"

// Constructor; uses RAII pattern to be valid after construction
nvidiamon::nvidiamon() : nvidia_stats{} {
  log_init(MONITOR_NAME);
#undef MONITOR_NAME
  for (const auto& param : params) {
    nvidia_stats.emplace(param.get_name(), prmon::monitored_value(param));
  }

  // Attempt to initialize NVML
  // If this works we are valid, but if not then we can't get data
  valid = init_nvml();
  last_seen_timestamp = 0;

  if (valid) {
    utilization.resize(max_samples);
    memory_info.resize(max_samples);
  }
}

// Destructor; shuts down NVML if it was initialized
nvidiamon::~nvidiamon() {
  if (valid) {
    nvmlReturn_t result = nvmlShutdown();
    if (result != NVML_SUCCESS) {
      warning("nvmlShutdown was not successfully finished");
    }
  }
}

void nvidiamon::update_stats(const std::vector<pid_t>& pids, const std::string read_path) {
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

    result = nvmlDeviceGetProcessUtilization(device, utilization.data(), &util_count, last_seen_timestamp);

    if (result == NVML_ERROR_INSUFFICIENT_SIZE) {
      warning("Utilization sample buffer size (" + std::to_string(max_samples) +
              ") exceeded. Consider increasing max_samples.");
      continue;
    }

    if (result == NVML_ERROR_NOT_FOUND) {
      util_count = 0;
    }

    if (result != NVML_SUCCESS && result != NVML_ERROR_NOT_FOUND) {
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
    }

    if (result != NVML_SUCCESS && result != NVML_ERROR_NOT_FOUND) {
      warning("Failed to get process memory utilization for GPU index " + std::to_string(gpu_idx) + 
              ": " + std::string(nvmlErrorString(result)));
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

// Return NVIDIA stats
prmon::monitored_value_map const nvidiamon::get_text_stats() {
  prmon::monitored_value_map nvidia_stat_map{};
  for (const auto& value : nvidia_stats) {
    nvidia_stat_map[value.first] = value.second.get_value();
  }
  return nvidia_stat_map;
}

// For JSON return the peaks
prmon::monitored_value_map const nvidiamon::get_json_total_stats() {
  prmon::monitored_value_map nvidia_stat_map{};
  for (const auto& value : nvidia_stats) {
    nvidia_stat_map[value.first] = value.second.get_max_value();
  }
  return nvidia_stat_map;
}

// And the averages
prmon::monitored_average_map const nvidiamon::get_json_average_stats(
    unsigned long long elapsed_clock_ticks) {
  prmon::monitored_average_map nvidia_stat_map{};
  for (const auto& value : nvidia_stats) {
    nvidia_stat_map[value.first] = value.second.get_average_value();
  }
  return nvidia_stat_map;
}

// Initialize NVML
bool nvidiamon::init_nvml() {
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

  ngpus = gpus;
  if (gpus == 0) {
    warning("NvmlInit() succeeded but no GPUs found");
    nvmlShutdown();
    return false;
  }
  return true;
}

// Return the parameter list
prmon::parameter_list const nvidiamon::get_parameter_list() { return params; }

// Collect related hardware information
void const nvidiamon::get_hardware_info(nlohmann::json& hw_json) {
  hw_json["HW"]["gpu"]["nGPU"] = ngpus;
  for (unsigned int i{0}; i < ngpus; ++i) {
    nvmlDevice_t device;
    nvmlReturn_t result;
    nvmlMemory_t memInfo;

    char name[NVML_DEVICE_NAME_BUFFER_SIZE] = {}; // NVML requires a char array

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
  return;
}

void const nvidiamon::get_unit_info(nlohmann::json& unit_json) {
  prmon::fill_units(unit_json, params);
  return;
}
#endif
