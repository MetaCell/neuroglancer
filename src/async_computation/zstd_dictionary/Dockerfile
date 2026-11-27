# 2.0.24 (2021-06-10)
FROM emscripten/emsdk@sha256:81ec54b7a096d28f24d906955dbf98ff336cca47658d980c243baa36f6484f9f

ENV ZSTD_1_5_7_SHA256 eb33e51f49a15e023950cd7825ca74a4a2b43db8354825ac24fc1b7ee09e6fa3

RUN mkdir -p /usr/src/zstd \
    && curl -SL -o /usr/src/zstd.tar.gz https://github.com/facebook/zstd/releases/download/v1.5.7/zstd-1.5.7.tar.gz
RUN echo "${ZSTD_1_5_7_SHA256} /usr/src/zstd.tar.gz" | sha256sum --check --status
RUN tar -xzC /usr/src/zstd -f /usr/src/zstd.tar.gz --strip-components=1 \
    && rm /usr/src/zstd.tar.gz
