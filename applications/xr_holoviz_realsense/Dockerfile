# syntax=docker/dockerfile:1

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

ARG BASE_IMAGE
FROM ${BASE_IMAGE} AS run

ARG DEBIAN_FRONTEND=noninteractive

# Install Vulkan SDK, ICD, lsb_release.
RUN wget -qO /etc/apt/sources.list.d/lunarg-vulkan-noble.list \
    http://packages.lunarg.com/vulkan/lunarg-vulkan-noble.list \
    && wget -qO- https://packages.lunarg.com/lunarg-signing-key-pub.asc | apt-key add -
RUN apt update \
    && apt install --no-install-recommends -y \
        vulkan-icd \
        vulkan-sdk \
        lsb-release
# Add librealsense apt source.
RUN curl -sSf https://librealsense.intel.com/Debian/librealsense.pgp | tee /etc/apt/keyrings/librealsense.pgp > /dev/null
RUN echo "deb [signed-by=/etc/apt/keyrings/librealsense.pgp] https://librealsense.intel.com/Debian/apt-repo `lsb_release -cs` main" \
    | tee /etc/apt/sources.list.d/librealsense.list
RUN rm -rf /var/lib/apt/lists/*


# Install OpenXR SDK Dependencies.
RUN apt update \
    && apt install --no-install-recommends -y \
        build-essential \
        cmake=3.31.* \
        cmake-data=3.31.* \
        libeigen3-dev \
        libgl1-mesa-dev \
        libvulkan-dev \
        libwayland-dev\
        libx11-xcb-dev \
        libxcb-dri2-0-dev \
        libxcb-glx0-dev \
        libxcb-icccm4-dev \
        libxcb-keysyms1-dev \
        libxcb-randr0-dev \
        libxrandr-dev \
        libxxf86vm-dev \
        mesa-common-dev \
        libopenxr-loader1 \
        libopenxr-dev \
        libopenxr1-monado \
        clang-format \
    && rm -rf /var/lib/apt/lists/*

# Install librealsense.
RUN apt update \
    && apt install --no-install-recommends -y \
        librealsense2-dev \
        libudev-dev \
        libusb-1.0-0-dev \
        libfastcdr-dev \
        libfastrtps-dev

# Install specific version of markupsafe for OpenXR build
RUN pip install markupsafe==2.0.1

# Install Python builder.
RUN pip install --upgrade pip
RUN pip install build

# Set the working directory
WORKDIR /app/xr_holoviz

#
# Holohub dev setup
#
RUN mkdir -p /tmp/scripts
COPY holohub /tmp/scripts/
RUN mkdir -p /tmp/scripts/utilities
COPY utilities /tmp/scripts/utilities/
RUN chmod +x /tmp/scripts/holohub
RUN /tmp/scripts/holohub setup && rm -rf /var/lib/apt/lists/*

# Enable autocomplete
RUN echo ". /etc/bash_completion.d/holohub_autocomplete" >> /etc/bash.bashrc