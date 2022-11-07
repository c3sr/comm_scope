#pragma once

struct UnaryData {
  char *ptr;
  hipEvent_t start;
  hipEvent_t stop;
  size_t pageSize;
  bool error;
};

