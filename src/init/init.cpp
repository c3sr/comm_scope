#include "init/init.hpp"

void myinit(int argc, char **argv) {
    std::cerr << "COMM_SCOPE IS HERE\n";
}

INIT(myinit);
