#include "DistributedLog.h"

namespace Logging {

std::ofstream* distributedLogFile = nullptr;

void InitLogging(std::string prefix) {
  int rank = 0;
  // MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  prefix.append(std::to_string(rank));
  prefix.append(".log");

  distributedLogFile = new std::ofstream(prefix, std::ios::out);
}

void StopLogging() { distributedLogFile->close(); }

}  // namespace Logging