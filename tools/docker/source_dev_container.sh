#!/usr/bin/env bash

#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#

# Run an Ascend development container against live vLLM and vllm-ascend trees.

set -euo pipefail

die() {
    echo "error: $*" >&2
    exit 1
}

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
if [[ -n ${VLLM_ASCEND_REPO:-} ]]; then
    ASCEND_REPO_INPUT=$VLLM_ASCEND_REPO
else
    ASCEND_REPO_INPUT=$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel 2>/dev/null) || \
        die "cannot locate the vllm-ascend checkout; set VLLM_ASCEND_REPO"
fi
[[ -d $ASCEND_REPO_INPUT ]] || \
    die "VLLM_ASCEND_REPO is not a directory: $ASCEND_REPO_INPUT"
ASCEND_REPO=$(cd -- "$ASCEND_REPO_INPUT" && pwd) || \
    die "cannot access VLLM_ASCEND_REPO: $ASCEND_REPO_INPUT"

NPU_DEVICES=${NPU_DEVICES:-0}
IMAGE=${IMAGE:-quay.io/ascend/vllm-ascend:v0.21.0rc1-openeuler}
SHM_SIZE=${SHM_SIZE:-4g}

VLLM_REMOTE=${VLLM_REMOTE:-https://github.com/vLLM-HUST/vllm-hust.git}
VLLM_REF=${VLLM_REF:-main}
VLLM_CACHE_DIR=${VLLM_CACHE_DIR:-${XDG_CACHE_HOME:-$HOME/.cache}/vllm-ascend-dev/vllm-hust}

MANAGED_LABEL=dev.vllm-hust.source-container
CACHE_MARKER=vllmAscend.sourceContainerCache

declare -a NPU_DEVICE_IDS=()
VLLM_SOURCE_MODE=
VLLM_SOURCE=
VLLM_SOURCE_REF=
VLLM_SHA=
IMAGE_ID=

parse_npu_devices() {
    local raw=$NPU_DEVICES
    local device_id
    local device_path
    local -A seen=()
    local -a requested_devices=()

    NPU_DEVICE_IDS=()
    if [[ $raw == "all" ]]; then
        while IFS= read -r device_path; do
            NPU_DEVICE_IDS+=("${device_path##*/davinci}")
        done < <(compgen -G '/dev/davinci[0-9]*' | sort -V)
        ((${#NPU_DEVICE_IDS[@]} > 0)) || die "no /dev/davinci<N> devices found"
        return
    fi

    IFS=',' read -ra requested_devices <<<"$raw"
    for device_id in "${requested_devices[@]}"; do
        device_id=${device_id//[[:space:]]/}
        [[ $device_id =~ ^[0-9]+$ ]] || \
            die "invalid NPU device '$device_id' in NPU_DEVICES=$raw"
        [[ -z ${seen[$device_id]+x} ]] || \
            die "duplicate NPU device $device_id in NPU_DEVICES=$raw"
        seen[$device_id]=1
        NPU_DEVICE_IDS+=("$device_id")
    done
    ((${#NPU_DEVICE_IDS[@]} > 0)) || die "NPU_DEVICES must not be empty"
}

device_set() {
    local IFS=,
    echo "${NPU_DEVICE_IDS[*]}"
}

default_container_name() {
    if [[ $NPU_DEVICES == "all" ]]; then
        echo "source-dev-npu-all"
        return
    fi
    local IFS=-
    echo "source-dev-npu${NPU_DEVICE_IDS[*]}"
}

container_name() {
    echo "${CONTAINER_NAME:-$(default_container_name)}"
}

require_ascend_source_tree() {
    [[ -f "$ASCEND_REPO/vllm_ascend/__init__.py" ]] || \
        die "VLLM_ASCEND_REPO is not a vllm-ascend source tree: $ASCEND_REPO"
}

prepare_vllm_source() {
    if [[ -n ${VLLM_REPO:-} ]]; then
        [[ -d $VLLM_REPO ]] || die "VLLM_REPO is not a directory: $VLLM_REPO"
        VLLM_SOURCE=$(cd -- "$VLLM_REPO" && pwd) || \
            die "cannot access VLLM_REPO: $VLLM_REPO"
        [[ -f "$VLLM_SOURCE/vllm/__init__.py" ]] || \
            die "VLLM_REPO is not a vLLM source tree: $VLLM_SOURCE"
        VLLM_SOURCE_MODE=local
        VLLM_SOURCE_REF=local
        VLLM_SHA=$(git -C "$VLLM_SOURCE" rev-parse HEAD 2>/dev/null) || \
            die "VLLM_REPO is not a Git checkout: $VLLM_SOURCE"
        return
    fi

    VLLM_SOURCE_MODE=managed-cache
    VLLM_SOURCE_REF=$VLLM_REF
    VLLM_SOURCE=$VLLM_CACHE_DIR
    mkdir -p -- "$(dirname -- "$VLLM_SOURCE")"
    if [[ ! -e $VLLM_SOURCE ]]; then
        git init --quiet "$VLLM_SOURCE"
        git -C "$VLLM_SOURCE" config --local "$CACHE_MARKER" true
        git -C "$VLLM_SOURCE" remote add origin "$VLLM_REMOTE"
    fi
    [[ -d $VLLM_SOURCE ]] || \
        die "VLLM_CACHE_DIR exists but is not a directory: $VLLM_SOURCE"
    VLLM_SOURCE=$(cd -- "$VLLM_SOURCE" && pwd)

    [[ -d "$VLLM_SOURCE/.git" ]] || \
        die "VLLM_CACHE_DIR exists but is not a Git repository: $VLLM_SOURCE"
    [[ $(git -C "$VLLM_SOURCE" config --local --get "$CACHE_MARKER" || true) == true ]] || \
        die "refusing to manage an unmarked checkout at VLLM_CACHE_DIR=$VLLM_SOURCE"
    [[ $(git -C "$VLLM_SOURCE" remote get-url origin) == "$VLLM_REMOTE" ]] || \
        die "cached origin does not match VLLM_REMOTE; choose a new VLLM_CACHE_DIR"
    [[ -z $(git -C "$VLLM_SOURCE" status --porcelain --untracked-files=all) ]] || \
        die "managed vLLM cache is dirty: $VLLM_SOURCE"

    echo "fetching $VLLM_REMOTE $VLLM_REF" >&2
    git -C "$VLLM_SOURCE" fetch --quiet --depth=1 origin "$VLLM_REF"
    VLLM_SHA=$(git -C "$VLLM_SOURCE" rev-parse FETCH_HEAD)

    # Do not change files under a running container.  Validation below will
    # request an explicit recreation when the managed branch has advanced.
    if container_exists && is_managed_container && \
        [[ $(container_label dev.vllm-hust.vllm-repo) == "$VLLM_SOURCE" ]] && \
        [[ $(container_label dev.vllm-hust.vllm-sha) != "$VLLM_SHA" ]]; then
        return
    fi

    git -C "$VLLM_SOURCE" -c advice.detachedHead=false \
        checkout --quiet --detach "$VLLM_SHA"
    [[ -f "$VLLM_SOURCE/vllm/__init__.py" ]] || \
        die "fetched ref is not a vLLM source tree: $VLLM_REMOTE@$VLLM_REF"
}

resolve_image() {
    if ! IMAGE_ID=$(docker image inspect --format '{{.Id}}' "$IMAGE" 2>/dev/null); then
        echo "pulling $IMAGE" >&2
        docker pull --quiet "$IMAGE" >/dev/null || die "failed to pull IMAGE=$IMAGE"
        IMAGE_ID=$(docker image inspect --format '{{.Id}}' "$IMAGE" 2>/dev/null) || \
            die "cannot resolve IMAGE=$IMAGE after pull"
    fi
    if [[ $IMAGE_ID =~ ^[0-9a-f]{64}$ ]]; then
        IMAGE_ID="sha256:$IMAGE_ID"
    fi
    [[ $IMAGE_ID == sha256:* ]] || die "Docker returned an invalid image ID for IMAGE=$IMAGE"
}

container_exists() {
    docker container inspect "$(container_name)" >/dev/null 2>&1
}

container_label() {
    docker container inspect \
        --format "{{index .Config.Labels \"$1\"}}" \
        "$(container_name)" 2>/dev/null || true
}

is_managed_container() {
    [[ $(container_label "$MANAGED_LABEL") == 1 ]]
}

require_managed_container() {
    container_exists || die "container does not exist: $(container_name)"
    is_managed_container || die \
        "container $(container_name) exists but is not managed by this script"
}

validate_existing_container() {
    require_managed_container
    [[ $(container_label dev.vllm-hust.vllm-repo) == "$VLLM_SOURCE" ]] || \
        die "existing container uses a different vLLM source; run recreate"
    [[ $(container_label dev.vllm-hust.vllm-ascend-repo) == "$ASCEND_REPO" ]] || \
        die "existing container uses a different vllm-ascend source; run recreate"
    [[ $(container_label dev.vllm-hust.npu-devices) == "$(device_set)" ]] || \
        die "existing container uses a different NPU set; run recreate"
    [[ $(container_label dev.vllm-hust.vllm-source-mode) == "$VLLM_SOURCE_MODE" ]] || \
        die "existing container uses a different vLLM source mode; run recreate"
    [[ $(container_label dev.vllm-hust.image-ref) == "$IMAGE" ]] || \
        die "existing container uses a different IMAGE reference; run recreate"
    [[ $(container_label dev.vllm-hust.image-id) == "$IMAGE_ID" ]] || \
        die "IMAGE resolved to a different image ID; run recreate"
    [[ $(docker container inspect --format '{{.Image}}' "$(container_name)") == "$IMAGE_ID" ]] || \
        die "existing container image provenance is inconsistent; run recreate"
    if [[ $VLLM_SOURCE_MODE == managed-cache ]]; then
        [[ $(container_label dev.vllm-hust.vllm-ref) == "$VLLM_SOURCE_REF" ]] || \
            die "existing container uses a different VLLM_REF; run recreate"
        [[ $(container_label dev.vllm-hust.vllm-sha) == "$VLLM_SHA" ]] || \
            die "vllm-hust:$VLLM_REF advanced; run recreate to use $VLLM_SHA"
    fi
}

start() {
    local name
    local device_id
    local device_path
    local -a device_args=()

    require_ascend_source_tree
    prepare_vllm_source
    name=$(container_name)

    for device_id in "${NPU_DEVICE_IDS[@]}"; do
        [[ -e /dev/davinci$device_id ]] || \
            die "NPU device does not exist: /dev/davinci$device_id"
        device_args+=(--device="/dev/davinci${device_id}")
    done
    for device_path in /dev/davinci_manager /dev/devmm_svm /dev/hisi_hdc; do
        [[ -e $device_path ]] || die "required Ascend device does not exist: $device_path"
    done
    [[ -d /usr/local/Ascend/driver ]] || \
        die "required Ascend driver directory does not exist: /usr/local/Ascend/driver"
    [[ -f /etc/ascend_install.info ]] || \
        die "required Ascend install file does not exist: /etc/ascend_install.info"
    resolve_image

    if container_exists; then
        validate_existing_container
        docker start "$name" >/dev/null
        echo "started existing container $name"
        return
    fi

    docker run -d \
        --name "$name" \
        --label "$MANAGED_LABEL=1" \
        --label "dev.vllm-hust.image-ref=$IMAGE" \
        --label "dev.vllm-hust.image-id=$IMAGE_ID" \
        --label "dev.vllm-hust.vllm-repo=$VLLM_SOURCE" \
        --label "dev.vllm-hust.vllm-ref=$VLLM_SOURCE_REF" \
        --label "dev.vllm-hust.vllm-sha=$VLLM_SHA" \
        --label "dev.vllm-hust.vllm-source-mode=$VLLM_SOURCE_MODE" \
        --label "dev.vllm-hust.vllm-ascend-repo=$ASCEND_REPO" \
        --label "dev.vllm-hust.npu-devices=$(device_set)" \
        "${device_args[@]}" \
        --device=/dev/davinci_manager \
        --device=/dev/devmm_svm \
        --device=/dev/hisi_hdc \
        --mount type=bind,src=/usr/local/Ascend/driver,dst=/usr/local/Ascend/driver,readonly \
        --mount type=bind,src=/etc/ascend_install.info,dst=/etc/ascend_install.info,readonly \
        --mount "type=bind,src=$VLLM_SOURCE,dst=/workspace/vllm,readonly" \
        --mount "type=bind,src=$ASCEND_REPO,dst=/workspace/vllm-ascend" \
        --mount "type=bind,src=$ASCEND_REPO,dst=/workspace/repo" \
        --env PYTHONPATH=/workspace/vllm:/workspace/vllm-ascend \
        --env PYTHONDONTWRITEBYTECODE=1 \
        --env "SOURCE_DEV_EXPECTED_NPU_COUNT=${#NPU_DEVICE_IDS[@]}" \
        --env "SOURCE_DEV_NPU_DEVICES=$(device_set)" \
        --workdir /workspace/vllm-ascend \
        --shm-size "$SHM_SIZE" \
        "$IMAGE_ID" sleep infinity >/dev/null

    echo "created $name"
    echo "  image:       $IMAGE@$IMAGE_ID"
    echo "  vLLM:        $VLLM_SOURCE@$VLLM_SHA -> /workspace/vllm (read-only)"
    echo "  vLLM Ascend: $ASCEND_REPO -> /workspace/vllm-ascend"
    echo "  NPUs:        $(device_set)"
}

verify() {
    require_managed_container
    docker exec -i "$(container_name)" python - <<'PY'
import inspect

import vllm
import vllm_ascend

vllm_path = inspect.getfile(vllm)
ascend_path = inspect.getfile(vllm_ascend)
assert vllm_path.startswith("/workspace/vllm/"), vllm_path
assert ascend_path.startswith("/workspace/vllm-ascend/"), ascend_path
print(f"vllm={vllm_path}")
print(f"vllm_ascend={ascend_path}")
print("PASS: both Python packages resolve from the live workspace")
PY
    echo "vllm_commit=$(container_label dev.vllm-hust.vllm-sha)"
}

verify_npu() {
    require_managed_container
    docker exec "$(container_name)" python -c '
import os

import torch
import torch_npu  # noqa: F401

expected = int(os.environ["SOURCE_DEV_EXPECTED_NPU_COUNT"])
actual = torch.npu.device_count()
assert actual == expected, f"expected {expected} visible NPUs, found {actual}"
for device_id in range(actual):
    torch.npu.set_device(device_id)
    value = torch.arange(8, device=f"npu:{device_id}")
    assert value.cpu().tolist() == list(range(8))
print(f"PASS: {actual} mounted NPU(s) are usable")
'
}

status() {
    if ! container_exists; then
        echo "$(container_name) does not exist"
        return
    fi
    docker container inspect --format \
        'name={{.Name}} status={{.State.Status}} image={{.Config.Image}}' \
        "$(container_name)"
    docker container inspect --format '{{range .Mounts}}{{println .Source "->" .Destination}}{{end}}' \
        "$(container_name)"
    echo "npu_devices=$(container_label dev.vllm-hust.npu-devices)"
    echo "image_ref=$(container_label dev.vllm-hust.image-ref)"
    echo "image_id=$(container_label dev.vllm-hust.image-id)"
    echo "vllm_source_mode=$(container_label dev.vllm-hust.vllm-source-mode)"
    echo "vllm_ref=$(container_label dev.vllm-hust.vllm-ref)"
    echo "vllm_sha=$(container_label dev.vllm-hust.vllm-sha)"
}

stop() {
    require_managed_container
    docker stop "$(container_name)"
}

remove() {
    container_exists || return
    require_managed_container
    docker rm -f "$(container_name)"
}

recreate() {
    remove
    start
}

usage() {
    cat <<EOF
Usage: $(basename "$0") {start|recreate|verify|verify-npu|status|shell|stop|remove}

Environment overrides:
  NPU_DEVICES       comma-separated physical IDs or "all" (default: 0)
  CONTAINER_NAME    container name (default: derived from NPU_DEVICES)
  IMAGE             base vllm-ascend image
  VLLM_REPO         optional local vLLM checkout; skips managed download
  VLLM_REMOTE       managed vLLM remote (default: vLLM-HUST/vllm-hust)
  VLLM_REF          managed vLLM branch/tag/commit (default: main)
  VLLM_CACHE_DIR    managed checkout location below XDG_CACHE_HOME
  VLLM_ASCEND_REPO  live vllm-ascend checkout (default: this repository)
  SHM_SIZE          container /dev/shm size (default: 4g)
EOF
}

main() {
    parse_npu_devices

    case ${1:-} in
        start) start ;;
        recreate) recreate ;;
        verify) verify ;;
        verify-npu) verify_npu ;;
        status) status ;;
        shell)
            require_managed_container
            exec docker exec -it "$(container_name)" bash
            ;;
        stop) stop ;;
        remove) remove ;;
        *) usage; exit 2 ;;
    esac
}

if [[ ${BASH_SOURCE[0]} == "$0" ]]; then
    main "$@"
fi
