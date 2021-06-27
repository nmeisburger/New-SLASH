#pragma once

// #include <mpi.h>

#include <fstream>
#include <iostream>

namespace Logging {

extern std::ofstream* distributedLogFile;

void InitLogging(std::string prefix);

void StopLogging();

#define LOG (*Logging::distributedLogFile) << "[" << __BASE_FILE__ << ":" << __LINE__ << "] "

}  // namespace Logging