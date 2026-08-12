/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
/*!
 * \file dispatch_ffn_tiling.cpp
 * \brief
 */
#include "vector"
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "tiling_base/error_log.h"
#include "hcom_topo_info.h"
#include "register/op_def_registry.h"
#include "dispatch_ffn_combine_bf16_tiling.h"
#include <vector>
#include <map>
#include <algorithm>
#include <cstdlib>
#include <limits>
#include <string>
#include "moe_init_routing_v2/moe_init_routing_v2_tiling.h"

using namespace AscendC;
using namespace ge;

namespace {
    const char *K_INNER_DEBUG = "DispatchFFNCombineBF16 Tiling Debug";
    constexpr uint32_t ATTR_GROUP_INDEX = 0;
    constexpr uint32_t ATTR_MAX_OUTPUT_SIZE_INDEX = 1;
    constexpr uint32_t ATTR_IS_TRANS_B = 2;
    constexpr uint32_t ATTR_WEIGHT_NZ = 3;
    constexpr uint64_t INIT_TILINGKEY = 1000000;
    constexpr uint64_t TILINGKEY_TRANS_B = 1U;
    constexpr uint64_t TILINGKEY_WEIGHT_NZ = 10;
    constexpr uint32_t X_INDEX = 0;
    constexpr uint32_t WEIGHT_INDEX = 1;
    constexpr uint32_t WEIGHT2_INDEX = 2;
    constexpr uint32_t EXPERTID_INDEX = 3;
    constexpr uint32_t PROBS_INDEX = 6;
    constexpr uint32_t X_ACTIVE_MASK_INDEX = 7;
    constexpr uint32_t OUT_INDEX = 0;
    constexpr uint32_t EXPERT_TOKEN_NUMS_INDEX = 1;
    constexpr uint32_t BLOCK_NUM = 20;
    constexpr uint32_t SYSTEM_NEED_WORKSPACE = 16 * 1024 * 1024;
    constexpr uint64_t MB_SIZE = 1024 * 1024UL;
    constexpr uint32_t MIN_WORLD_SIZE = 2;
    constexpr uint32_t MAX_WORLD_SIZE = 8;
    constexpr uint32_t MIN_LOCAL_EXPERTS = 3;
    constexpr uint32_t MAX_LOCAL_EXPERTS = 64;
    constexpr uint32_t MAX_TOP_K = 8;
    constexpr uint32_t MAX_LOCAL_TOKENS = 512;
    constexpr uint32_t MAX_EXPERT_NUM = 5120;
    constexpr uint32_t COUNT_CONTROL_TAIL_SIZE = 2 * MB_SIZE;
    constexpr uint32_t COUNT_MATRIX_REGION_SIZE = MB_SIZE;
    constexpr uint32_t CONTROL_REGION_SIZE = MB_SIZE;
    constexpr uint32_t PER_TOKEN_SCALE_REGION_SIZE = MB_SIZE;
    constexpr uint32_t WINDOW_ALIGNMENT = 512;
}

namespace optiling {

static uint64_t GetConfiguredWindowSize()
{
    uint64_t windowSizeMiB = 200;
    const char *configured = getenv("HCCL_BUFFSIZE");
    if (configured == nullptr) {
        return windowSizeMiB * MB_SIZE;
    }

    try {
        const std::string value(configured);
        size_t parsedChars = 0;
        windowSizeMiB = std::stoull(value, &parsedChars);
        if (parsedChars != value.size() || windowSizeMiB > std::numeric_limits<uint16_t>::max()) {
            OP_LOGW(K_INNER_DEBUG, "Invalid HCCL_BUFFSIZE=%s; use the 200 MiB default.", configured);
            windowSizeMiB = 200;
        }
    } catch (const std::exception &e) {
        OP_LOGW(K_INNER_DEBUG, "Cannot parse HCCL_BUFFSIZE=%s (%s); use the 200 MiB default.", configured, e.what());
        windowSizeMiB = 200;
    }
    return windowSizeMiB * MB_SIZE;
}

static ge::graphStatus DispatchFFNCombineBF16CheckPhysicalDomain(
    gert::TilingContext *context, const DispatchFFNCombineBF16Info &info)
{
    const char *nodeName = context->GetNodeName();
    const uint64_t routeRows = static_cast<uint64_t>(info.M) * info.topK;
    const uint64_t globalExperts = static_cast<uint64_t>(info.expertPerRank) * info.worldSize;
    const uint64_t alignedGlobalExperts = (globalExperts + 1 + 127) / 128 * 128;
    const uint64_t countBytes = static_cast<uint64_t>(info.worldSize) * alignedGlobalExperts * sizeof(int32_t);
    // init-routing expands every local token into top-k routed rows inside
    // this rank's peer window.
    const uint64_t inputBytes = routeRows * info.K * sizeof(int16_t);
    const uint64_t outputBytes = static_cast<uint64_t>(info.maxOutputSize) * info.K * sizeof(int16_t);
    const uint64_t windowBytes = GetConfiguredWindowSize();
    const uint64_t scaleOffset = ((windowBytes / 3 + WINDOW_ALIGNMENT - 1) / WINDOW_ALIGNMENT) * WINDOW_ALIGNMENT;
    const uint64_t outputOffset = scaleOffset + PER_TOKEN_SCALE_REGION_SIZE;
    const uint64_t countOffset =
        windowBytes >= COUNT_CONTROL_TAIL_SIZE ? windowBytes - COUNT_CONTROL_TAIL_SIZE : 0;

    OP_TILING_CHECK(info.worldSize < MIN_WORLD_SIZE || info.worldSize > MAX_WORLD_SIZE,
        OP_LOGE(nodeName, "BF16 dispatch_ffn_combine requires EP in [%u, %u], got %u.",
            MIN_WORLD_SIZE, MAX_WORLD_SIZE, info.worldSize), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(info.expertPerRank < MIN_LOCAL_EXPERTS || info.expertPerRank > MAX_LOCAL_EXPERTS,
        OP_LOGE(nodeName, "BF16 dispatch_ffn_combine requires local experts in [%u, %u], got %u.",
            MIN_LOCAL_EXPERTS, MAX_LOCAL_EXPERTS, info.expertPerRank), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(globalExperts + 1 > MAX_EXPERT_NUM,
        OP_LOGE(nodeName, "BF16 dispatch_ffn_combine global expert domain %lu exceeds route limit %u.",
            globalExperts + 1, MAX_EXPERT_NUM), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(info.M == 0 || info.M > MAX_LOCAL_TOKENS,
        OP_LOGE(nodeName, "BF16 dispatch_ffn_combine local tokens must be in [1, %u], got %u.",
            MAX_LOCAL_TOKENS, info.M), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(info.topK == 0 || info.topK > MAX_TOP_K,
        OP_LOGE(nodeName, "BF16 dispatch_ffn_combine top-k must be in [1, %u], got %u.",
            MAX_TOP_K, info.topK), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(info.K == 0 || info.N == 0 || info.K % 256 != 0 || info.N % 256 != 0,
        OP_LOGE(nodeName, "BF16 dispatch_ffn_combine requires K and gate/up width N to be positive multiples of 256, got K=%u N=%u.",
            info.K, info.N), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(info.topK > globalExperts,
        OP_LOGE(nodeName, "BF16 dispatch_ffn_combine top-k=%u exceeds global experts=%lu.",
            info.topK, globalExperts), return ge::GRAPH_FAILED);
    const uint64_t requiredReceiverRows = routeRows * info.worldSize;
    OP_TILING_CHECK(info.maxOutputSize == 0 || info.maxOutputSize < requiredReceiverRows,
        OP_LOGE(nodeName,
            "BF16 dispatch_ffn_combine max_output_size=%u is smaller than worst-case receiver rows=%lu.",
            info.maxOutputSize, requiredReceiverRows), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(windowBytes <= COUNT_CONTROL_TAIL_SIZE || countBytes > COUNT_MATRIX_REGION_SIZE,
        OP_LOGE(nodeName, "BF16 dispatch_ffn_combine count state requires %lu bytes inside a %u-byte region.",
            countBytes, COUNT_MATRIX_REGION_SIZE), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(inputBytes > scaleOffset,
        OP_LOGE(nodeName, "BF16 dispatch_ffn_combine input requires %lu bytes before peer-scale offset %lu.",
            inputBytes, scaleOffset), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(outputOffset > countOffset || outputBytes > countOffset - outputOffset,
        OP_LOGE(nodeName, "BF16 dispatch_ffn_combine output requires %lu bytes, but HCCL_BUFFSIZE=%lu MiB leaves %lu bytes.",
            outputBytes, windowBytes / MB_SIZE, outputOffset <= countOffset ? countOffset - outputOffset : 0),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(countOffset + countBytes > windowBytes - CONTROL_REGION_SIZE,
        OP_LOGE(nodeName, "BF16 dispatch_ffn_combine HCCL window control/count regions overlap."),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

static int32_t CeilDev(int32_t num, int32_t div)
{
    if (div == 0) {
        return 0;
    }
    return (num + div - 1) / div;
}

static ge::graphStatus DispatchFFNCombineBF16CheckAttrAndSetTiling(gert::TilingContext *context, DispatchFFNCombineBF16Info& info)
{
    auto attrs = context->GetAttrs();
    OP_TILING_CHECK(attrs == nullptr, OP_LOGE(K_INNER_DEBUG, "attrs is null."), return ge::GRAPH_FAILED);

    auto groupPtr = attrs->GetAttrPointer<char>(static_cast<int>(ATTR_GROUP_INDEX));
    auto maxOutputSizePtr = attrs->GetAttrPointer<int>(ATTR_MAX_OUTPUT_SIZE_INDEX);
    auto is_trans_b = attrs->GetAttrPointer<bool>(ATTR_IS_TRANS_B);
    auto weight_nz = attrs->GetAttrPointer<bool>(ATTR_WEIGHT_NZ);
    OP_TILING_CHECK(groupPtr == nullptr || strlen(groupPtr) == 0,
    OP_LOGE(K_INNER_DEBUG, "group is invalid."), return GRAPH_FAILED);
    OP_TILING_CHECK(maxOutputSizePtr == nullptr || *maxOutputSizePtr <= 0,
        OP_LOGE(K_INNER_DEBUG, "max_output_size must be positive."), return GRAPH_FAILED);

    OP_TILING_CHECK(is_trans_b == nullptr,
        OP_LOGE(K_INNER_DEBUG, "is_trans_b is invalid."), return GRAPH_FAILED);
    OP_TILING_CHECK(weight_nz == nullptr,
        OP_LOGE(K_INNER_DEBUG, "weight_nz is invalid."), return GRAPH_FAILED);

    info.maxOutputSize = *maxOutputSizePtr;
    info.isTransposeB = *is_trans_b;
    info.isWeightNz = *weight_nz;

    int64_t rankSize;
    (void)ge::HcomTopoInfo::Instance().GetGroupRankSize(groupPtr, rankSize);
    info.worldSize = rankSize;

    OP_LOGD(K_INNER_DEBUG, "maxOutputSize=%d ", info.maxOutputSize);
    OP_LOGD(K_INNER_DEBUG, "rankSize=%d ", info.worldSize);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus DispatchFFNCombineBF16CheckShapeAndSetTiling(gert::TilingContext *context, DispatchFFNCombineBF16Info &info)
{
    const char *nodeName = context->GetNodeName();

    const gert::StorageShape *aStorageShape = context->GetInputShape(X_INDEX);
    auto expertIdxTensor = context->GetDynamicInputTensor(EXPERTID_INDEX, 0);
    const gert::StorageShape *probsShape = context->GetInputShape(PROBS_INDEX);
    const gert::StorageShape *outShape = context->GetOutputShape(OUT_INDEX);
    const gert::StorageShape *expertTokenNumsShape = context->GetOutputShape(EXPERT_TOKEN_NUMS_INDEX);
    OP_TILING_CHECK(aStorageShape == nullptr || aStorageShape->GetStorageShape().GetDimNum() != 2,
        OP_LOGE(nodeName, "a must be a rank-2 tensor."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(expertIdxTensor == nullptr || expertIdxTensor->GetStorageShape().GetDimNum() != 2,
        OP_LOGE(nodeName, "expert_idx must be a rank-2 tensor."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(probsShape == nullptr || probsShape->GetStorageShape().GetDimNum() != 2,
        OP_LOGE(nodeName, "probs must be a rank-2 tensor."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(outShape == nullptr || outShape->GetStorageShape().GetDimNum() != 2,
        OP_LOGE(nodeName, "out must be a rank-2 tensor."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(expertTokenNumsShape == nullptr || expertTokenNumsShape->GetStorageShape().GetDimNum() != 1,
        OP_LOGE(nodeName, "expert_token_nums must be a rank-1 tensor."), return ge::GRAPH_FAILED);
    uint32_t M = aStorageShape->GetStorageShape().GetDim(0);
    uint32_t K = aStorageShape->GetStorageShape().GetDim(1);

    auto wTensor = context->GetDynamicInputTensor(WEIGHT_INDEX, 0);
    auto w2Tensor = context->GetDynamicInputTensor(WEIGHT2_INDEX, 0);
    OP_TILING_CHECK(wTensor == nullptr || w2Tensor == nullptr,
        OP_LOGE(nodeName, "w1 and w2 must each contain at least one tensor."), return ge::GRAPH_FAILED);
    uint32_t wTensorDims = wTensor->GetOriginShape().GetDimNum();
    OP_TILING_CHECK(wTensorDims == 0 || wTensor->GetStorageShape().GetDimNum() < wTensorDims,
        OP_LOGE(nodeName, "w1 has an invalid storage/origin shape."), return ge::GRAPH_FAILED);
    uint32_t N = wTensor->GetStorageShape().GetDim(wTensorDims - 1);

    uint32_t topK = expertIdxTensor->GetStorageShape().GetDim(1);
    OP_TILING_CHECK(expertIdxTensor->GetStorageShape().GetDim(0) != M ||
        probsShape->GetStorageShape().GetDim(0) != M || probsShape->GetStorageShape().GetDim(1) != topK,
        OP_LOGE(nodeName, "expert_idx and probs must both have shape [M, top_k]."),
        return ge::GRAPH_FAILED);
    uint32_t listLen = 0;
    while (true) {
        auto wTensorT = context->GetDynamicInputTensor(WEIGHT_INDEX, ++listLen);
        if (wTensorT == nullptr) {break;}
    }

    uint32_t expertPerRank;
    if (listLen == 1) {
        expertPerRank = wTensor->GetStorageShape().GetDim(0);
    } else {
        expertPerRank = listLen;
    }
    OP_TILING_CHECK(outShape->GetStorageShape().GetDim(0) != M ||
        outShape->GetStorageShape().GetDim(1) != K,
        OP_LOGE(nodeName, "out must have shape [M, K]."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(expertTokenNumsShape->GetStorageShape().GetDim(0) != expertPerRank,
        OP_LOGE(nodeName, "expert_token_nums length must equal local experts=%u.", expertPerRank),
        return ge::GRAPH_FAILED);

    info.M = M;
    info.N = N;
    info.K = K;
    info.expertPerRank = expertPerRank;
    info.topK = topK;
    info.listLen = listLen;
    OP_LOGD(K_INNER_DEBUG, "M=%d ", info.M);
    OP_LOGD(K_INNER_DEBUG, "K=%d ", info.K);
    OP_LOGD(K_INNER_DEBUG, "N=%d ", info.N);
    OP_LOGD(K_INNER_DEBUG, "expertPerRank=%d ", info.expertPerRank);
    OP_LOGD(K_INNER_DEBUG, "topK=%d ", info.topK);
    OP_LOGD(K_INNER_DEBUG, "listLen=%d ", info.listLen);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus DispatchFFNCombineBF16GetPlatformInfoAndSetTiling(gert::TilingContext *context, DispatchFFNCombineBF16Info& info)
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t aivNum = ascendcPlatform.GetCoreNumAiv();
    uint64_t ubSize = 0U;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    info.aivNum = aivNum;
    info.totalUbSize = ubSize;

    OP_LOGD(K_INNER_DEBUG, "aivNum=%d", info.aivNum);
    OP_LOGD(K_INNER_DEBUG, "ubSize=%lu", info.totalUbSize);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckXActiveMaskShape(
    gert::TilingContext *context, const char *nodeName, const DispatchFFNCombineBF16Info &info)
{
    const gert::StorageShape *maskShape = context->GetOptionalInputShape(X_ACTIVE_MASK_INDEX);
    if (maskShape == nullptr) {
        return ge::GRAPH_SUCCESS;
    }

    const gert::Shape &storageShape = maskShape->GetStorageShape();
    OP_TILING_CHECK(storageShape.GetDimNum() != 1,
        OP_LOGE(nodeName, "x_active_mask must be rank 1, got rank %lu.", storageShape.GetDimNum()),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(storageShape.GetDim(0) != static_cast<int64_t>(info.M),
        OP_LOGE(nodeName, "x_active_mask length must equal local tokens M=%u, got %ld.",
            info.M, storageShape.GetDim(0)), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

void SetTilingData(CoCTiling &cocTilingData, DispatchFFNCombineBF16Info &info)
{
    cocTilingData.m0 = 128;
    cocTilingData.k0 = 256;
    cocTilingData.n0 = 256;
    cocTilingData.swizzleDirect = 1;
    cocTilingData.swizzleOffset = 7;
    cocTilingData.ubMoveNum = 16 * 1024;
    cocTilingData.pValue = 1;
    cocTilingData.commNpuSplit = info.worldSize;
    cocTilingData.commDataSplit = 1;
    cocTilingData.lenPerLoop = cocTilingData.m0 * cocTilingData.n0 / 2;
}

static ge::graphStatus DispatchFFNCombineBF16TilingFuncImpl(gert::TilingContext *context)
{
    const char *nodeName = context->GetNodeName();
    OP_LOGI(nodeName, "Enter DispatchFFNCombineBF16 tiling func.");

    // 1. tilingData
    DispatchFFNCombineBF16TilingData *tilingData = context->GetTilingData<DispatchFFNCombineBF16TilingData>();
    OP_TILING_CHECK(tilingData == nullptr, OP_LOGE(nodeName, "tilingData is nullptr."),
        return ge::GRAPH_FAILED);
    OP_LOGI(nodeName, "DispatchFFNCombineBF16 get tilingData.");
    DispatchFFNCombineBF16Info& info = tilingData->dispatchFFNCombineBF16Info;
    OP_LOGI(nodeName, "DispatchFFNCombineBF16 get tilingData info.");

    OP_TILING_CHECK(DispatchFFNCombineBF16CheckAttrAndSetTiling(context, info) != ge::GRAPH_SUCCESS,
        OP_LOGE(context->GetNodeName(), "DispatchFFNCombineBF16 CheckAttrAndSetTiling Failed"),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(DispatchFFNCombineBF16CheckShapeAndSetTiling(context, info) != ge::GRAPH_SUCCESS,
        OP_LOGE(context->GetNodeName(), "DispatchFFNCombineBF16 CheckShapeAndSetTiling Failed"),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(DispatchFFNCombineBF16GetPlatformInfoAndSetTiling(context, info) != ge::GRAPH_SUCCESS,
        OP_LOGE(context->GetNodeName(), "DispatchFFNCombineBF16 GetPlatformInfoAndSetTiling Failed"),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(CheckXActiveMaskShape(context, nodeName, info) != ge::GRAPH_SUCCESS,
        OP_LOGE(nodeName, "DispatchFFNCombineBF16 x_active_mask validation failed"),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(DispatchFFNCombineBF16CheckPhysicalDomain(context, info) != ge::GRAPH_SUCCESS,
        OP_LOGE(context->GetNodeName(), "DispatchFFNCombineBF16 physical-domain validation failed"),
        return ge::GRAPH_FAILED);

    SetTilingData(tilingData->cocTiling, info);

    // 2. set blockDim
    uint32_t blockDim = 1U;
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    auto aicNum = ascendcPlatform.GetCoreNumAic();
    auto aivNum = ascendcPlatform.GetCoreNumAiv();
    blockDim = ascendcPlatform.CalcTschBlockDim(aivNum, aicNum, aivNum);
    context->SetBlockDim(blockDim);

    // 3. set tilingKey
    uint64_t tilingKey = INIT_TILINGKEY;
    tilingKey += info.isTransposeB ? TILINGKEY_TRANS_B : 0;
    tilingKey += info.isWeightNz ? TILINGKEY_WEIGHT_NZ : 0;
    context->SetTilingKey(tilingKey);

    OP_LOGD(K_INNER_DEBUG, "tilingKey=%d", tilingKey);

    optiling::MoeInitRoutingV2TilingBase moeInitRoutingQuantV2TilingBase;
    int64_t inuptXDtypeSize = sizeof(int16_t);
    int64_t scaleDim0 = 0;
    int64_t ubSize = 196352;
    int64_t expertCapacity = 0;
    // The final ID is a sentinel used only for inactive graph-padding rows.
    int64_t expertNum = info.expertPerRank * info.worldSize + 1;
    int64_t activeNum = info.M * info.topK;
    int64_t dropPadMode = 0;
     int64_t expertTokensCountOrCumsumFlag = 2;
     bool expertTokensBeforeCapacityFlag = false;
     int64_t quantMode = 1;
     uint32_t aivNumInitRouting = 2 * BLOCK_NUM;
    moeInitRoutingQuantV2TilingBase.DoTiling(info.M, info.K, info.topK, expertCapacity, expertNum, activeNum, dropPadMode,
        expertTokensCountOrCumsumFlag, expertTokensBeforeCapacityFlag, inuptXDtypeSize, quantMode, scaleDim0, aivNumInitRouting, ubSize);
    uint64_t initRoutingQuantTilingKey = moeInitRoutingQuantV2TilingBase.tilingKey_;
    size_t initRoutingWorkspace = moeInitRoutingQuantV2TilingBase.workspaceSize_;

    tilingData->cocTiling.moeInitRoutingQuantV2TilingData = moeInitRoutingQuantV2TilingBase.moeInitRoutingTilingData;
    tilingData->cocTiling.moeInitRoutingQuantV2TilingData.vbsComputeParamsOp = moeInitRoutingQuantV2TilingBase.moeInitRoutingTilingData.vbsComputeParamsOp;
    tilingData->cocTiling.moeInitRoutingQuantV2TilingData.vmsMiddleComputeParamsOp = moeInitRoutingQuantV2TilingBase.moeInitRoutingTilingData.vmsMiddleComputeParamsOp;
    tilingData->cocTiling.moeInitRoutingQuantV2TilingData.sortOutComputeParamsOp = moeInitRoutingQuantV2TilingBase.moeInitRoutingTilingData.sortOutComputeParamsOp;
    tilingData->cocTiling.moeInitRoutingQuantV2TilingData.srcToDstComputeParamsOp = moeInitRoutingQuantV2TilingBase.moeInitRoutingTilingData.srcToDstComputeParamsOp;
    tilingData->cocTiling.moeInitRoutingQuantV2TilingData.srcToDstCapacityComputeParamsOp = moeInitRoutingQuantV2TilingBase.moeInitRoutingTilingData.srcToDstCapacityComputeParamsOp;
    tilingData->cocTiling.moeInitRoutingQuantV2TilingData.gatherOutComputeParamsOp = moeInitRoutingQuantV2TilingBase.moeInitRoutingTilingData.gatherOutComputeParamsOp;
    tilingData->cocTiling.initRoutingQuantTilingKey = initRoutingQuantTilingKey;
    // OP_LOGE(initRoutingTilingKey, " initRoutingTilingKey.");
    OP_LOGD(K_INNER_DEBUG, "tilingKey=%ld", initRoutingQuantTilingKey);

    // 4. workspace
    size_t *workSpaces = context->GetWorkspaceSizes(1);
    OP_TILING_CHECK(workSpaces == nullptr, OP_LOGE(nodeName, "workSpaces is nullptr."),
        return ge::GRAPH_FAILED);

    uint32_t n2 = info.K;
    uint32_t k2 = info.N / 2;

    const uint64_t expandedRowIdxWorkspace =
        static_cast<uint64_t>((info.M + 256 - 1) / 256) * 256 * info.topK * sizeof(int32_t);
    const uint64_t expertIdxScratchWorkspace =
        (static_cast<uint64_t>(info.M) * info.topK * sizeof(int32_t) + 32 - 1) / 32 * 32;
    const uint64_t countMatrixWorkspace =
        static_cast<uint64_t>(info.worldSize) * info.worldSize * info.expertPerRank * sizeof(int32_t);
    uint64_t cocWorkspace = expandedRowIdxWorkspace +
                            std::max(expertIdxScratchWorkspace, countMatrixWorkspace) +
                            countMatrixWorkspace * 2 +
                            static_cast<uint64_t>(info.maxOutputSize) * sizeof(float) * 2 +
                            static_cast<uint64_t>(info.maxOutputSize) * info.N * sizeof(int16_t) +
                            static_cast<uint64_t>(info.maxOutputSize) * n2 * sizeof(int16_t) +
                            static_cast<uint64_t>(info.maxOutputSize) * info.K * sizeof(int16_t) +
                            static_cast<uint64_t>(info.maxOutputSize) * k2 * sizeof(int16_t) +
                            static_cast<uint64_t>(info.worldSize) * sizeof(int32_t) * 16 +
                            static_cast<uint64_t>(info.expertPerRank + info.worldSize) * sizeof(int32_t) * 16;
                            // std::max(info.maxOutputSize * info.N * sizeof(int16_t), info.maxOutputSize * n2 * sizeof(int16_t)) +
                            // std::max(info.maxOutputSize * info.K * sizeof(int8_t), info.maxOutputSize * k2 * sizeof(int8_t));

    const uint64_t routingWorkspace =
        expandedRowIdxWorkspace + expertIdxScratchWorkspace + initRoutingWorkspace;
    workSpaces[0] = SYSTEM_NEED_WORKSPACE + std::max(cocWorkspace, routingWorkspace);


    // 5. communication
    auto attrs = context->GetAttrs();
    auto group = attrs->GetAttrPointer<char>(static_cast<int>(ATTR_GROUP_INDEX));
    uint32_t opType = 8U;
    std::string algConfig = "AlltoAll=level0:fullmesh;level1:pairwise";
    AscendC::Mc2CcTilingConfig mc2CcTilingConfig(group, opType, algConfig);
    mc2CcTilingConfig.GetTiling(tilingData->mc2InitTiling);
    mc2CcTilingConfig.GetTiling(tilingData->mc2CcTiling);

    OP_LOGI(nodeName, "Leave DispatchFFNCombineBF16 tiling func.");
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus DispatchFFNCombineBF16TilingFunc(gert::TilingContext* context)
{
    return DispatchFFNCombineBF16TilingFuncImpl(context);
}

struct DispatchFFNCombineBF16CompileInfo {};
ge::graphStatus TilingParseForDispatchFFNCombineBF16(gert::TilingParseContext *context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(DispatchFFNCombineBF16)
    .Tiling(DispatchFFNCombineBF16TilingFunc)
    .TilingParse<DispatchFFNCombineBF16CompileInfo>(TilingParseForDispatchFFNCombineBF16);
} // namespace optiling
