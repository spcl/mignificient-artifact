#include <chrono>
#include <iomanip>
#include <iostream>
#include <nvml.h>
#include <stdexcept>
#include <string>
#include <cstdlib> 
#include <array>

enum {
  UNKNOWN,
  VERSION_1,
  VERSION_2
} version;

void check_nvml_error(nvmlReturn_t result, const std::string &error_message) {
  if (result != NVML_SUCCESS) {
    throw std::runtime_error(error_message + ": " + nvmlErrorString(result));
  }
}

void initialize_nvml() {
  nvmlReturn_t result = nvmlInit();
  check_nvml_error(result, "Failed to initialize NVML");
}

unsigned int get_device_count() {
  unsigned int device_count;
  nvmlReturn_t result = nvmlDeviceGetCount(&device_count);
  check_nvml_error(result, "Failed to get device count");
  return device_count;
}

nvmlDevice_t select_device(unsigned int device_id) {
  nvmlDevice_t device;
  nvmlReturn_t result = nvmlDeviceGetHandleByIndex(device_id, &device);
  check_nvml_error(result, "Failed to get device handle");
  return device;
}

struct DeviceUtilization {
  unsigned int gpu;
  unsigned int memory;
  unsigned long long memoryUsed;
  unsigned long long memoryFree;
  unsigned long long memoryReserved;
  unsigned long long memoryTotal;
};

DeviceUtilization get_fast_device_utilization(nvmlDevice_t device) {
  DeviceUtilization util;

  // Get GPU and Memory utilization rates
  nvmlUtilization_t utilization;
  nvmlReturn_t result = nvmlDeviceGetUtilizationRates(device, &utilization);
  check_nvml_error(result, "Failed to get GPU utilization");
  util.gpu = utilization.gpu;
  util.memory = utilization.memory;

  // Check the version
  if (version == UNKNOWN) {
    nvmlMemory_v2_t memory;
    memory.version = NVML_STRUCT_VERSION(Memory, 2);
    result = nvmlDeviceGetMemoryInfo_v2(device, &memory);  
    if (result == NVML_SUCCESS) {
      version = VERSION_2;
    } else {
      version = VERSION_1;
    }
  }

  // Get the memory info according to the version
  if (version == VERSION_1) {
    nvmlMemory_t memory;
    result = nvmlDeviceGetMemoryInfo(device, &memory);
    util.memoryUsed = memory.used;
    util.memoryFree = memory.free;
    util.memoryReserved = 0;
    util.memoryTotal = memory.total;
  } else {
    nvmlMemory_v2_t memory;
    memory.version = NVML_STRUCT_VERSION(Memory, 2);
    result = nvmlDeviceGetMemoryInfo_v2(device, &memory);
    util.memoryUsed = memory.used;
    util.memoryFree = memory.free;
    util.memoryReserved = memory.reserved;
    util.memoryTotal = memory.total;
  }

  return util;
}

// Function to parse output and extract GPU utilization
void get_slow_device_utilization(int device_id) {
  // Command to execute
  std::string cmd = "nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.total,memory.free,memory.used,memory.reserved --format=csv,noheader,nounits -lms=1 --id=" + std::to_string(device_id);
    
  // Open the command as a pipe
  FILE* pipe = popen(cmd.c_str(), "r");
  if (!pipe) {
      std::cerr << "Error opening pipe!\n";
      return;
  }

  char buffer[256];
  DeviceUtilization util;
  while (true) {
    if (fgets(buffer, sizeof(buffer), pipe) == nullptr) {
      std::cerr << "Error: Failed to read from command output" << std::endl;
      break;  // Exit loop if the pipe fails
    }

    // Get high-resolution timestamp in nanoseconds
    auto now = std::chrono::system_clock::now();
    auto nanoseconds = std::chrono::duration_cast<std::chrono::nanoseconds>(now.time_since_epoch()).count();

    // Convert buffer to string and parse CSV output
    std::stringstream ss(buffer);
    std::string token;
    
    try {
      std::getline(ss, token, ','); util.gpu = std::stoi(token);
      std::getline(ss, token, ','); util.memory = std::stoi(token);
      std::getline(ss, token, ','); util.memoryTotal = std::stoull(token);
      std::getline(ss, token, ','); util.memoryFree = std::stoull(token);
      std::getline(ss, token, ','); util.memoryUsed = std::stoull(token);
      std::getline(ss, token, ','); util.memoryReserved = std::stoull(token);
    } catch (const std::exception& e) {
      std::cerr << "Error parsing output: " << e.what() << " output: " << buffer << std::endl;
      continue; // Skip this iteration if parsing fails
    }

    // Print the parsed data with timestamp
    std::cout << nanoseconds << "," 
              << util.gpu << "," 
              << util.memory << "," 
              << util.memoryTotal << ","
              << util.memoryFree << ","
              << util.memoryUsed << ","
              << util.memoryReserved << "\n";
  }

  pclose(pipe);
}

int main(int argc, char *argv[]) {
  try {
    if (argc != 2) {
      std::cerr << "Usage: " << argv[0] << " <GPU_ID>\n";
      return 1;
    }

    unsigned int device_id = std::stoi(argv[1]);

    initialize_nvml();

    unsigned int device_count = get_device_count();
    if (device_count == 0) {
      std::cout << "No NVIDIA GPUs found.\n";
      return 0;
    }

    if (device_id >= device_count) {
      std::cerr << "Invalid GPU ID. Please provide a number between 0 and "
                << device_count - 1 << ".\n";
      return 1;
    }

    nvmlDevice_t device = select_device(device_id);
    uint MIG, pending_mode;
    nvmlReturn_t result = nvmlDeviceGetMigMode(device, &MIG, &pending_mode);
    if (result != NVML_SUCCESS) {
      std::cerr << "Failed to get MIG mode: " << nvmlErrorString(result) << std::endl;
      MIG = 0;
    } 

    // Print header in CSV format
    std::cout << "time,GPU,mem,memTotal,memFree,memUsed,memReserved\n";

    if (MIG) {
      get_slow_device_utilization(device_id);
    } else {
      while (true) {
        auto now = std::chrono::system_clock::now();
        auto nanoseconds = std::chrono::duration_cast<std::chrono::nanoseconds>(now.time_since_epoch()).count();
        DeviceUtilization util = get_fast_device_utilization(device);
        std::cout << nanoseconds << "," 
                  << util.gpu << "," 
                  << util.memory << "," 
                  << (util.memoryTotal / (1024 * 1024)) << ","
                  << (util.memoryFree / (1024 * 1024)) << ","
                  << (util.memoryUsed / (1024 * 1024)) << ","
                  << (util.memoryReserved / (1024 * 1024)) << "\n";
      }
    }

    result = nvmlShutdown();
    check_nvml_error(result, "Failed to shut down NVML");
  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
  }

  return 0;
}


