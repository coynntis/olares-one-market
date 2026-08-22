# DeepSeek Harness runtime — Node + agent tools only.
# Do NOT npm-install @deepseek-ai/dsh here. Chart boots
#   npx -y @deepseek-ai/dsh@${DSH_VERSION}
# so harness upgrades are a Settings env change, not an image rebuild.
FROM docker.io/library/node:22.19-bookworm

ENV DEBIAN_FRONTEND=noninteractive \
    NODE_ENV=production

# APT::Sandbox::User=root — qemu user-mode (amd64 image on arm64 host) breaks apt GPG otherwise
RUN apt-get -o APT::Sandbox::User=root update \
    && apt-get -o APT::Sandbox::User=root install -y --no-install-recommends \
      git \
      python3 \
      python3-pip \
      python3-venv \
      curl \
      jq \
      unzip \
      ca-certificates \
      build-essential \
      ripgrep \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /data \
    && chown -R 1000:1000 /data

USER 1000
WORKDIR /data
