// Standalone probe: does cuTENSOR 2.6 block-sparse contract support batch modes?
// Computes D[b,m,n] = A[b,m,k] * B[b,k,n] with b as an outermost batch mode.
//
// Layout: each tensor has one non-zero block; modes all have 1 section.
// If 2.6 supports batch modes natively, this call should succeed and produce
// the correct result. Previous versions (2.4.1-2.5.0) returned NOT_SUPPORTED.

#include <cutensor.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cmath>

#define CK(x) do { auto _s=(x); if(_s!=cudaSuccess){fprintf(stderr,"CUDA %s\n",cudaGetErrorString(_s));std::exit(1);} } while(0)
#define CT(x) do { auto _s=(x); if(_s!=CUTENSOR_STATUS_SUCCESS){fprintf(stderr,"cuTENSOR [%s] %d: %s\n",#x,(int)_s,cutensorGetErrorString(_s));std::exit(2);} } while(0)

int main() {
    size_t ver = cutensorGetVersion();
    printf("cuTENSOR version: %zu (major=%zu minor=%zu patch=%zu)\n",
           ver, ver/10000, (ver/100)%100, ver%100);

    cutensorHandle_t handle;
    CT(cutensorCreate(&handle));

    // Sanity: try non-batched matmul first to confirm setup
    {
        cutensorBlockSparseTensorDescriptor_t a,b,d;
        uint32_t spm[]={1,1};
        int32_t nz[]={0,0};
        int64_t eA[]={3,4}, eB[]={4,5}, eD[]={3,5};
        CT(cutensorCreateBlockSparseTensorDescriptor(handle,&a,2,1,spm,eA,nz,nullptr,CUDA_R_32F));
        CT(cutensorCreateBlockSparseTensorDescriptor(handle,&b,2,1,spm,eB,nz,nullptr,CUDA_R_32F));
        CT(cutensorCreateBlockSparseTensorDescriptor(handle,&d,2,1,spm,eD,nz,nullptr,CUDA_R_32F));
        int32_t mA[]={'m','k'}, mB[]={'k','n'}, mD[]={'m','n'};
        cutensorOperationDescriptor_t od;
        auto s = cutensorCreateBlockSparseContraction(handle,&od,
            a,mA,CUTENSOR_OP_IDENTITY, b,mB,CUTENSOR_OP_IDENTITY,
            d,mD,CUTENSOR_OP_IDENTITY, d,mD, CUTENSOR_COMPUTE_DESC_32F);
        printf("[sanity 2D matmul] create -> %d (%s)\n",(int)s,cutensorGetErrorString(s));
        if (s==CUTENSOR_STATUS_SUCCESS) cutensorDestroyOperationDescriptor(od);
        cutensorDestroyBlockSparseTensorDescriptor(a);
        cutensorDestroyBlockSparseTensorDescriptor(b);
        cutensorDestroyBlockSparseTensorDescriptor(d);
    }

    // Dims: b=2, m=3, k=4, n=5
    const int B=2, M=3, K=4, N=5;

    // --- A[b,m,k] ---
    cutensorBlockSparseTensorDescriptor_t descA;
    {
        uint32_t numModes = 3;
        uint32_t sectionsPerMode[] = {1,1,1};
        int64_t extent[] = {B, M, K};
        int32_t nzCoord[]  = {0,0,0};
        CT(cutensorCreateBlockSparseTensorDescriptor(
            handle, &descA, numModes, /*nnz=*/1,
            sectionsPerMode, extent, nzCoord, /*stride=*/nullptr, CUDA_R_32F));
    }
    // --- B[b,k,n] ---
    cutensorBlockSparseTensorDescriptor_t descB;
    {
        uint32_t numModes = 3;
        uint32_t sectionsPerMode[] = {1,1,1};
        int64_t extent[] = {B, K, N};
        int32_t nzCoord[]  = {0,0,0};
        CT(cutensorCreateBlockSparseTensorDescriptor(
            handle, &descB, numModes, 1,
            sectionsPerMode, extent, nzCoord, nullptr, CUDA_R_32F));
    }
    // --- D[b,m,n] (and C aliased) ---
    cutensorBlockSparseTensorDescriptor_t descD;
    {
        uint32_t numModes = 3;
        uint32_t sectionsPerMode[] = {1,1,1};
        int64_t extent[] = {B, M, N};
        int32_t nzCoord[]  = {0,0,0};
        CT(cutensorCreateBlockSparseTensorDescriptor(
            handle, &descD, numModes, 1,
            sectionsPerMode, extent, nzCoord, nullptr, CUDA_R_32F));
    }

    // Modes: 'b'=98,'m'=109,'k'=107,'n'=110
    int32_t modeA[] = {'b','m','k'};
    int32_t modeB[] = {'b','k','n'};
    int32_t modeD[] = {'b','m','n'};

    cutensorOperationDescriptor_t opDesc;
    cutensorStatus_t cs = cutensorCreateBlockSparseContraction(
        handle, &opDesc,
        descA, modeA, CUTENSOR_OP_IDENTITY,
        descB, modeB, CUTENSOR_OP_IDENTITY,
        descD, modeD, CUTENSOR_OP_IDENTITY,
        descD, modeD,
        CUTENSOR_COMPUTE_DESC_32F);
    printf("cutensorCreateBlockSparseContraction -> %d (%s)\n",
           (int)cs, cutensorGetErrorString(cs));
    if (cs != CUTENSOR_STATUS_SUCCESS) {
        printf("RESULT: batch mode NOT supported at create-contraction stage\n");
        return 0;
    }

    cutensorPlanPreference_t pref;
    CT(cutensorCreatePlanPreference(handle, &pref, CUTENSOR_ALGO_DEFAULT, CUTENSOR_JIT_MODE_NONE));
    uint64_t wsSize = 0;
    CT(cutensorEstimateWorkspaceSize(handle, opDesc, pref, CUTENSOR_WORKSPACE_DEFAULT, &wsSize));
    printf("workspace size: %lu\n", (unsigned long)wsSize);

    cutensorPlan_t plan;
    cs = cutensorCreatePlan(handle, &plan, opDesc, pref, wsSize);
    printf("cutensorCreatePlan -> %d (%s)\n", (int)cs, cutensorGetErrorString(cs));
    if (cs != CUTENSOR_STATUS_SUCCESS) {
        printf("RESULT: batch mode NOT supported at plan stage\n");
        return 0;
    }

    // Host data: column-major per block (default stride=nullptr)
    std::vector<float> hA(B*M*K), hB(B*K*N), hD(B*M*N,0.f);
    for (size_t i=0;i<hA.size();++i) hA[i] = 0.01f*(float)(i+1);
    for (size_t i=0;i<hB.size();++i) hB[i] = 0.02f*(float)(i+1);

    float *dA,*dB,*dD,*dWS=nullptr;
    CK(cudaMalloc(&dA, hA.size()*4));
    CK(cudaMalloc(&dB, hB.size()*4));
    CK(cudaMalloc(&dD, hD.size()*4));
    if (wsSize) CK(cudaMalloc(&dWS, wsSize));
    CK(cudaMemcpy(dA, hA.data(), hA.size()*4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dB, hB.data(), hB.size()*4, cudaMemcpyHostToDevice));
    CK(cudaMemset(dD, 0, hD.size()*4));

    const void* Aptrs[1] = {dA};
    const void* Bptrs[1] = {dB};
    const void* Cptrs[1] = {dD};
    void* Dptrs[1]       = {dD};
    float alpha = 1.f, beta = 0.f;
    cs = cutensorBlockSparseContract(handle, plan,
        &alpha, Aptrs, Bptrs,
        &beta,  Cptrs, Dptrs,
        dWS, wsSize, /*stream=*/0);
    printf("cutensorBlockSparseContract -> %d (%s)\n", (int)cs, cutensorGetErrorString(cs));
    CK(cudaDeviceSynchronize());
    if (cs != CUTENSOR_STATUS_SUCCESS) {
        printf("RESULT: batch mode NOT supported at execute stage\n");
        return 0;
    }

    CK(cudaMemcpy(hD.data(), dD, hD.size()*4, cudaMemcpyDeviceToHost));

    // Reference (column-major: index(b,m,n) = b + B*m + B*M*n, since modes are
    // listed [b,m,n] with default stride=1,B,B*M). Same for A,B.
    auto idxA = [&](int b,int m,int k){ return b + B*m + B*M*k; };
    auto idxB = [&](int b,int k,int n){ return b + B*k + B*K*n; };
    auto idxD = [&](int b,int m,int n){ return b + B*m + B*M*n; };
    double maxerr = 0;
    int nbad = 0;
    for (int b=0;b<B;++b)
     for (int m=0;m<M;++m)
      for (int n=0;n<N;++n) {
        double ref = 0;
        for (int k=0;k<K;++k) ref += (double)hA[idxA(b,m,k)]*(double)hB[idxB(b,k,n)];
        double got = hD[idxD(b,m,n)];
        double err = std::fabs(got-ref);
        if (err > 1e-4) ++nbad;
        if (err > maxerr) maxerr = err;
      }
    printf("max abs err = %g, mismatches = %d / %d\n", maxerr, nbad, B*M*N);
    if (nbad == 0) printf("RESULT: batch mode SUPPORTED and CORRECT\n");
    else           printf("RESULT: call succeeded but result INCORRECT (batch likely treated as free/contract mode)\n");

    cutensorDestroyPlan(plan);
    cutensorDestroyPlanPreference(pref);
    cutensorDestroyOperationDescriptor(opDesc);
    cutensorDestroyBlockSparseTensorDescriptor(descA);
    cutensorDestroyBlockSparseTensorDescriptor(descB);
    cutensorDestroyBlockSparseTensorDescriptor(descD);
    cutensorDestroy(handle);
    return 0;
}
