FROM c3sr/scope:amd64-cuda92-latest

RUN apt-get update && apt-get install -y --no-install-recommends --no-install-suggests \
    libnuma-dev \
    && rm -rf /var/lib/apt/lists/*

RUN cd build \
  && cmake -DENABLE_COMM=1 .. \
  && make