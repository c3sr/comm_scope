#include "scope/init/logger.hpp"
#include "init/version.hpp"

#include <string>
#include <iostream>

std::string comm_scope::version() {

    std::string refspec = SCOPE_GIT_REFSPEC;
    std::string hash = SCOPE_GIT_HASH;
    std::string local_changes;
    if (std::string("DIRTY") == std::string(SCOPE_GIT_LOCAL_CHANGES)) {
        local_changes = "-dirty";
    } else {
        local_changes = "";
    }

    if (refspec.rfind("refs/heads/", 0) == 0) {
        return refspec.substr(11, refspec.size() - 11) + std::string("-") + hash + local_changes;
    } else if (refspec.rfind("refs/tags/", 0) == 0) {
        return refspec.substr(10, refspec.size() - 10) + local_changes;
    } else {
      LOG(debug, "refspec={}", refspec);
      LOG(debug, "hash={}", hash);
      return std::string("unknown");
    }

}
