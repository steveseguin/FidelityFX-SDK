"""CPU-only regression for ParallelSort::Execute's direct/indirect dispatch paths.

Run from any directory: python3 tests/parallel_sort_dispatch.py
Requires a C++17 compiler (CXX, default g++). No Windows SDK or GPU is needed.
The real class declaration, CPU sort helpers and complete Execute method are
compiled with recording renderer doubles. Init's resource state is modeled;
this does not test resource creation, shader execution or sorted GPU output.
"""

import os
from pathlib import Path
import re
import subprocess
import tempfile


ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "Kits/Cauldron2/dx12/rendermodules/gpuparticle"

SUPPORT = r'''
#include <cassert>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>
#define FFX_CPU 1
#define FFX_DIVIDE_ROUNDING_UP(a, b) (((a) + (b) - 1) / (b))
using FfxUInt32 = uint32_t;
using FfxInt32 = int32_t;
#include "parallelsort/ffx_parallelsort.h"
using ParallelSortConstants = FfxParallelSortConstants;
#define FFX_PARALLELSORT_ITERATION_COUNT (32u / FFX_PARALLELSORT_SORT_BITS_PER_PASS)
struct SetupIndirectCB { uint32_t maxThreadGroups, shift; };
#define ASSERT_CRITICAL 0
#define CauldronAssert(level, condition, message) assert(condition)
namespace cauldron {
struct CommandList {};
struct RootSignature {};
struct PipelineObject {};
struct IndirectWorkload {};
struct BufferAddressInfo {};
struct Buffer {
    const Buffer* GetResource() const { return this; }
    BufferAddressInfo GetAddressInfo() const { return {}; }
};
struct ParameterSet {
    unsigned srvs = 0, uavs = 0, binds = 0;
    const Buffer* lastSrv = nullptr;
    const Buffer* slots[5] = {};
    virtual ~ParameterSet() = default;
    virtual void SetBufferSRV(const Buffer* b, unsigned) {
        assert(b); ++srvs; lastSrv = b;
    }
    virtual void SetBufferUAV(const Buffer* b, unsigned slot) {
        assert(b && slot < 5); ++uavs; slots[slot] = b;
    }
    void UpdateRootConstantBuffer(const BufferAddressInfo*, unsigned) {}
    void Bind(CommandList*, PipelineObject*) { ++binds; }
};
enum class ResourceState { UnorderedAccess, IndirectArgument, ConstantBufferResource };
struct Barrier {
    static Barrier UAV(const Buffer* p) { assert(p); return {}; }
    static Barrier Transition(const Buffer* p, ResourceState, ResourceState) { assert(p); return {}; }
};
std::vector<unsigned> dispatches;
unsigned indirectCalls = 0;
std::vector<ParallelSortConstants> constants;
struct DynamicBufferPool {
    BufferAddressInfo AllocConstantBuffer(size_t bytes, const void* p) {
        if (bytes == sizeof(ParallelSortConstants))
            constants.push_back(*static_cast<const ParallelSortConstants*>(p));
        else
            assert(bytes == sizeof(SetupIndirectCB));
        return {};
    }
};
DynamicBufferPool* GetDynamicBufferPool() { static DynamicBufferPool pool; return &pool; }
void SetPipelineState(CommandList*, PipelineObject*) {}
void ResourceBarrier(CommandList*, int, const Barrier*) {}
void Dispatch(CommandList*, unsigned x, unsigned y, unsigned z) {
    assert(y == 1 && z == 1); dispatches.push_back(x);
}
void ExecuteIndirect(CommandList*, IndirectWorkload* workload, const Buffer* p, unsigned count, unsigned offset) {
    assert(workload && p && count == 1 && offset == 0); ++indirectCalls;
}
}
using namespace cauldron;
'''

TESTS = r'''
// Init and destruction require the real renderer. The fixture mirrors the
// mode-specific allocation in Init, preserving null indirect state in direct mode.
ParallelSort::~ParallelSort() {}
void check(bool indirect, bool payload) {
    ParallelSort sort;
    Buffer keys, values, scratchKeys, scratchValues, sums, reduced, args, reduceArgs, cb, count;
    ParameterSet setup, passes[5][FFX_PARALLELSORT_ITERATION_COUNT];
    IndirectWorkload workload;
    sort.m_bIndirect = indirect;
    sort.m_bHasPayload = payload;
    sort.m_pSortScratchBuffer = &scratchKeys;
    sort.m_pPayloadScratchBuffer = payload ? &scratchValues : nullptr;
    sort.m_pScratchBuffer = &sums;
    sort.m_pReducedScratchBuffer = &reduced;
    if (indirect) {
        sort.m_pSetupIndirectArgsParameters = &setup;
        sort.m_pIndirectWorkload = &workload;
        sort.m_pIndirectCountScatterArgsBuffer = &args;
        sort.m_pIndirectReduceScanArgsBuffer = &reduceArgs;
        sort.m_pIndirectConstantBuffer = &cb;
    }
    for (unsigned i = 0; i < FFX_PARALLELSORT_ITERATION_COUNT; ++i) {
        sort.m_pCountParameters[i] = &passes[0][i];
        sort.m_pCountReduceParameters[i] = &passes[1][i];
        sort.m_pScanParameters[i] = &passes[2][i];
        sort.m_pScanAddParameters[i] = &passes[3][i];
        sort.m_pScatterParameters[i] = &passes[4][i];
    }
    NumKeys num{};
    if (indirect) num.pNumKeysBuffer = &count;
    else num.numKeys = 513;
    for (unsigned repeat = 0; repeat < 2; ++repeat) {
        dispatches.clear(); indirectCalls = 0; constants.clear();
        sort.Execute(nullptr, num, &keys, payload ? &values : nullptr);
        assert(dispatches.size() == (indirect ? 16 : 40));
        assert(indirectCalls == (indirect ? 32 : 0));
        assert(constants.size() == (indirect ? 0 : 8));
        for (unsigned i = 0; i < 8; ++i) {
            if (!indirect) {
                assert(constants[i].numKeys == 513 && constants[i].shift == 4 * i);
                const unsigned expected[] = {2, 16, 1, 16, 2};
                for (unsigned p = 0; p < 5; ++p) assert(dispatches[5*i+p] == expected[p]);
            }
            else assert(dispatches[2*i] == 1 && dispatches[2*i+1] == 1);
            for (unsigned p = 0; p < 5; ++p) assert(passes[p][i].binds == repeat + 1);
            assert(passes[4][i].slots[0] == (i % 2 ? &scratchKeys : &keys));
            assert(passes[4][i].slots[1] == (i % 2 ? &keys : &scratchKeys));
            assert(passes[4][i].uavs == (repeat + 1) * (payload ? 5 : 3));
            if (payload) {
                assert(passes[4][i].slots[3] == (i % 2 ? &scratchValues : &values));
                assert(passes[4][i].slots[4] == (i % 2 ? &values : &scratchValues));
            }
        }
        assert(setup.srvs == (indirect ? repeat + 1 : 0));
        assert(setup.uavs == (indirect ? 3 * (repeat + 1) : 0));
        if (indirect) {
            assert(setup.lastSrv == &count && setup.slots[0] == &cb);
            assert(setup.slots[1] == &args && setup.slots[2] == &reduceArgs);
        }
    }
}
int main(int argc, char** argv) {
    assert(argc == 3);
    check(argv[1][0] == '1', argv[2][0] == '1');
    std::cout << "PASS mode=" << argv[1] << " payload=" << argv[2] << " repeated twice\n";
}
'''


def main():
    header = (MODULE / "parallelsort.h").read_text()
    header = re.sub(r'^#(?:include .*|pragma once)$', '', header, flags=re.MULTILINE)
    source = (MODULE / "parallelsort.cpp").read_text()
    execute = source[source.index("void ParallelSort::Execute("):]
    unit = SUPPORT + "\n#define private public\n" + header
    unit += "\n#undef private\n" + execute + TESTS
    with tempfile.TemporaryDirectory() as directory:
        cpp = Path(directory) / "dispatch.cpp"
        binary = Path(directory) / "dispatch"
        cpp.write_text(unit)
        subprocess.run([os.environ.get("CXX", "g++"), "-std=c++17", "-O1", "-g",
                        "-fsanitize=undefined", "-fno-sanitize-recover=all",
                        "-I", str(MODULE), str(cpp), "-o", str(binary)], check=True)
        failures = 0
        for indirect, payload in [(False, False), (False, True), (True, False), (True, True)]:
            result = subprocess.run([str(binary), str(int(indirect)), str(int(payload))])
            failures += result.returncode != 0
        if failures:
            raise SystemExit(f"{failures} dispatch modes failed")


if __name__ == "__main__":
    main()
