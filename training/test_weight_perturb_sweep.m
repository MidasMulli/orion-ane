// test_weight_perturb_sweep.m — Main 24 Agent 2
// Perturbation sensitivity sweep using disk-patch + unload/reload path.
// Forked from test_weight_reload.m. One conv kernel, single-element weight
// perturbations across magnitudes [+0.1, +0.5, +1.0, +2.0, +10.0].
//
// Per trial: read original FP16 weight, write perturbed value to weights/weight.bin,
// unload, reload, eval, compute max_diff vs baseline, restore, reload again, verify
// clean revert.

#import <Foundation/Foundation.h>
#import <objc/runtime.h>
#import <objc/message.h>
#import <dlfcn.h>
#import <IOSurface/IOSurface.h>
#import <mach/mach_time.h>
#include <math.h>
#include <stdio.h>

static mach_timebase_info_data_t g_tb;
static double tb_ms(uint64_t t) { return (double)t * g_tb.numer / g_tb.denom / 1e6; }

static IOSurfaceRef make_surface(size_t bytes) {
    return IOSurfaceCreate((__bridge CFDictionaryRef)@{
        (id)kIOSurfaceWidth:@(bytes), (id)kIOSurfaceHeight:@1,
        (id)kIOSurfaceBytesPerElement:@1, (id)kIOSurfaceBytesPerRow:@(bytes),
        (id)kIOSurfaceAllocSize:@(bytes), (id)kIOSurfacePixelFormat:@0});
}

// Build weight blob: 128-byte header + fp16 payload at offset 128, BLOBFILE offset=64
static NSData *build_weight_blob(_Float16 *w, int rows, int cols) {
    int ws = rows * cols * 2;
    int tot = 128 + ws;
    uint8_t *b = (uint8_t*)calloc(tot, 1);
    b[0] = 1; b[4] = 2;
    b[64] = 0xEF; b[65] = 0xBE; b[66] = 0xAD; b[67] = 0xDE; b[68] = 1;
    *(uint32_t*)(b+72) = ws;
    *(uint32_t*)(b+80) = 128;
    memcpy(b + 128, w, ws);
    return [NSData dataWithBytesNoCopy:b length:tot freeWhenDone:YES];
}

static NSString *gen_mil(int ch, int sp) {
    return [NSString stringWithFormat:
        @"program(1.3)\n"
        "[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}, "
        "{\"coremlc-version\", \"3505.4.1\"}, {\"coremltools-component-milinternal\", \"\"}, "
        "{\"coremltools-version\", \"9.0\"}})]\n"
        "{\n"
        "    func main<ios18>(tensor<fp32, [1, %d, 1, %d]> x) {\n"
        "        string pt = const()[name=string(\"pt\"), val=string(\"valid\")];\n"
        "        tensor<int32, [2]> st = const()[name=string(\"st\"), val=tensor<int32, [2]>([1,1])];\n"
        "        tensor<int32, [4]> pd = const()[name=string(\"pd\"), val=tensor<int32, [4]>([0,0,0,0])];\n"
        "        tensor<int32, [2]> dl = const()[name=string(\"dl\"), val=tensor<int32, [2]>([1,1])];\n"
        "        int32 gr = const()[name=string(\"gr\"), val=int32(1)];\n"
        "        string to16 = const()[name=string(\"to16\"), val=string(\"fp16\")];\n"
        "        tensor<fp16, [1,%d,1,%d]> x16 = cast(dtype=to16,x=x)[name=string(\"cin\")];\n"
        "        tensor<fp16, [%d,%d,1,1]> W = const()[name=string(\"W\"), "
        "val=tensor<fp16, [%d,%d,1,1]>(BLOBFILE(path=string(\"@model_path/weights/weight.bin\"), offset=uint64(64)))];\n"
        "        tensor<fp16, [1,%d,1,%d]> y16 = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=W,x=x16)"
        "[name=string(\"conv\")];\n"
        "        string to32 = const()[name=string(\"to32\"), val=string(\"fp32\")];\n"
        "        tensor<fp32, [1,%d,1,%d]> y = cast(dtype=to32,x=y16)[name=string(\"cout\")];\n"
        "    } -> (y);\n"
        "}\n", ch, sp, ch, sp, ch, ch, ch, ch, ch, sp, ch, sp];
}

int main() { @autoreleasepool {
    setbuf(stdout, NULL);
    mach_timebase_info(&g_tb);
    dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine", RTLD_NOW);

    Class g_D  = NSClassFromString(@"_ANEInMemoryModelDescriptor");
    Class g_I  = NSClassFromString(@"_ANEInMemoryModel");
    Class g_AR = NSClassFromString(@"_ANERequest");
    Class g_AIO= NSClassFromString(@"_ANEIOSurfaceObject");
    if (!g_D || !g_I || !g_AR || !g_AIO) { printf("FAIL: ANE classes\n"); return 1; }

    int CH = 64, SP = 32;
    int N = CH * CH;

    // Baseline weights: dense pseudo-random fp16, deterministic seed
    _Float16 *W0 = (_Float16*)calloc(N, sizeof(_Float16));
    for (int i = 0; i < N; i++) {
        // simple LCG, range ~[-0.5,0.5]
        unsigned u = (i*2654435761u) ^ 0xdeadbeef;
        float f = ((u % 1000) / 1000.0f) - 0.5f;
        W0[i] = (_Float16)f;
    }
    NSData *wdata0 = build_weight_blob(W0, CH, CH);

    NSString *mil = gen_mil(CH, SP);
    NSDictionary *weights = @{@"@model_path/weights/weight.bin": @{@"offset":@0, @"data":wdata0}};
    NSData *milData = [mil dataUsingEncoding:NSUTF8StringEncoding];

    // Compile + load
    uint64_t t0 = mach_absolute_time();
    id desc = ((id(*)(Class,SEL,id,id,id))objc_msgSend)(g_D, @selector(modelWithMILText:weights:optionsPlist:), milData, weights, nil);
    id mdl = ((id(*)(Class,SEL,id))objc_msgSend)(g_I, @selector(inMemoryModelWithDescriptor:), desc);
    id hx = ((id(*)(id,SEL))objc_msgSend)(mdl, @selector(hexStringIdentifier));
    NSString *td = [NSTemporaryDirectory() stringByAppendingPathComponent:hx];
    NSFileManager *fm = [NSFileManager defaultManager];
    [fm createDirectoryAtPath:[td stringByAppendingPathComponent:@"weights"] withIntermediateDirectories:YES attributes:nil error:nil];
    [milData writeToFile:[td stringByAppendingPathComponent:@"model.mil"] atomically:YES];
    NSString *weightPath = [td stringByAppendingPathComponent:@"weights/weight.bin"];
    [wdata0 writeToFile:weightPath atomically:YES];

    NSError *e = nil;
    BOOL ok = ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(mdl, @selector(compileWithQoS:options:error:), 21, @{}, &e);
    if (!ok) { printf("FAIL compile\n"); return 1; }
    ok = ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(mdl, @selector(loadWithQoS:options:error:), 21, @{}, &e);
    if (!ok) { printf("FAIL load\n"); return 1; }
    double compile_ms = tb_ms(mach_absolute_time() - t0);
    printf("Compile+load: %.1fms\n", compile_ms);
    printf("tmpDir: %s\n", [td UTF8String]);
    printf("weightPath: %s\n", [weightPath UTF8String]);

    int inBytes = CH * SP * 4, outBytes = CH * SP * 4;
    IOSurfaceRef ioIn = make_surface(inBytes);
    IOSurfaceRef ioOut = make_surface(outBytes);

    // Helper to (re)build request
    id (^mkreq)(void) = ^id() {
        id wI = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_AIO, @selector(objectWithIOSurface:), ioIn);
        id wO = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_AIO, @selector(objectWithIOSurface:), ioOut);
        return ((id(*)(Class,SEL,id,id,id,id,id,id,id))objc_msgSend)(g_AR,
            @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:),
            @[wI], @[@0], @[wO], @[@0], nil, nil, @0);
    };

    // Fixed input
    IOSurfaceLock(ioIn, 0, NULL);
    float *inp = (float*)IOSurfaceGetBaseAddress(ioIn);
    for (int c = 0; c < CH; c++)
        for (int s = 0; s < SP; s++)
            inp[c*SP+s] = (float)(c*SP + s + 1) * 0.01f;
    IOSurfaceUnlock(ioIn, 0, NULL);

    // === Baseline ===
    id req = mkreq();
    NSError *ee = nil;
    ok = ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(mdl, @selector(evaluateWithQoS:options:request:error:), 21, @{}, req, &ee);
    if (!ok) { printf("FAIL baseline eval\n"); return 1; }

    float *baseline = (float*)malloc(outBytes);
    IOSurfaceLock(ioOut, kIOSurfaceLockReadOnly, NULL);
    memcpy(baseline, IOSurfaceGetBaseAddress(ioOut), outBytes);
    IOSurfaceUnlock(ioOut, kIOSurfaceLockReadOnly, NULL);

    float bmin=baseline[0], bmax=baseline[0];
    for (int i = 0; i < CH*SP; i++) { if (baseline[i]<bmin) bmin=baseline[i]; if (baseline[i]>bmax) bmax=baseline[i]; }
    printf("Baseline range: [%.4f, %.4f] span=%.4f\n", bmin, bmax, bmax-bmin);
    printf("Baseline[0..3]: [%.6f %.6f %.6f %.6f]\n", baseline[0], baseline[1], baseline[2], baseline[3]);

    // Save baseline
    FILE *fp = fopen("/tmp/main24_weight_perturb_outputs/baseline.bin", "wb");
    fwrite(baseline, 1, outBytes, fp); fclose(fp);

    // === Perturbation sweep ===
    // Pick weight element: row 0, col 0 (= W0[0]). It contributes to output channel 0.
    int patch_idx = 0;
    float orig_f = (float)W0[patch_idx];
    printf("\nPerturb element idx=%d, original=%.6f\n", patch_idx, orig_f);

    float mags[5] = {0.1f, 0.5f, 1.0f, 2.0f, 10.0f};
    const char *mag_names[5] = {"+0.1","+0.5","+1.0","+2.0","+10.0"};

    // Output JSON
    FILE *jfp = fopen("/tmp/main24_weight_perturb_table.json", "w");
    fprintf(jfp, "{\n  \"meta\": {\n");
    fprintf(jfp, "    \"kernel\": \"conv %dx%d sp=%d, fp16, identity-bias-free\",\n", CH, CH, SP);
    fprintf(jfp, "    \"patch_idx\": %d,\n    \"original_value\": %.6f,\n", patch_idx, orig_f);
    fprintf(jfp, "    \"baseline_min\": %.6f, \"baseline_max\": %.6f, \"baseline_span\": %.6f,\n",
            bmin, bmax, bmax-bmin);
    fprintf(jfp, "    \"compile_load_ms\": %.3f,\n", compile_ms);
    fprintf(jfp, "    \"note\": \"disk-patch + unloadWithQoS + loadWithQoS + evaluateWithQoS path\"\n");
    fprintf(jfp, "  },\n  \"trials\": [\n");

    double reload_lat[5];
    for (int t = 0; t < 5; t++) {
        float mag = mags[t];
        _Float16 perturbed = (_Float16)(orig_f + mag);
        printf("\n=== Trial %d: %s (perturbed=%.6f) ===\n", t+1, mag_names[t], (float)perturbed);

        // Patch weight
        W0[patch_idx] = perturbed;
        NSData *wdataP = build_weight_blob(W0, CH, CH);
        [wdataP writeToFile:weightPath atomically:YES];

        // Unload
        ee = nil;
        uint64_t tu = mach_absolute_time();
        ((BOOL(*)(id,SEL,unsigned int,NSError**))objc_msgSend)(mdl, @selector(unloadWithQoS:error:), 21, &ee);
        // Reload
        uint64_t tr = mach_absolute_time();
        BOOL rok = ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(mdl, @selector(loadWithQoS:options:error:), 21, @{}, &ee);
        uint64_t te = mach_absolute_time();
        double unload_ms = tb_ms(tr - tu);
        double load_ms   = tb_ms(te - tr);
        reload_lat[t] = unload_ms + load_ms;
        printf("  unload=%.2fms load=%.2fms total=%.2fms ok=%d\n", unload_ms, load_ms, reload_lat[t], rok);

        // Re-create request, re-write input (it might have been clobbered? safer)
        req = mkreq();
        IOSurfaceLock(ioIn, 0, NULL);
        inp = (float*)IOSurfaceGetBaseAddress(ioIn);
        for (int c = 0; c < CH; c++)
            for (int s = 0; s < SP; s++)
                inp[c*SP+s] = (float)(c*SP + s + 1) * 0.01f;
        IOSurfaceUnlock(ioIn, 0, NULL);

        // Eval
        ee = nil;
        ok = ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(mdl, @selector(evaluateWithQoS:options:request:error:), 21, @{}, req, &ee);
        if (!ok) { printf("  FAIL eval: %s\n", ee?[[ee description] UTF8String]:"?"); }

        float *outp = (float*)malloc(outBytes);
        IOSurfaceLock(ioOut, kIOSurfaceLockReadOnly, NULL);
        memcpy(outp, IOSurfaceGetBaseAddress(ioOut), outBytes);
        IOSurfaceUnlock(ioOut, kIOSurfaceLockReadOnly, NULL);

        float max_diff = 0; int n_changed = 0; int n_nan = 0; int n_inf = 0;
        for (int i = 0; i < CH*SP; i++) {
            float d = fabsf(outp[i] - baseline[i]);
            if (isnan(outp[i])) n_nan++;
            if (isinf(outp[i])) n_inf++;
            if (d > max_diff) max_diff = d;
            if (d > 1e-6f) n_changed++;
        }
        printf("  max_diff=%.6f n_changed=%d/2048 nan=%d inf=%d\n", max_diff, n_changed, n_nan, n_inf);
        printf("  out[0..3]: [%.6f %.6f %.6f %.6f]\n", outp[0], outp[1], outp[2], outp[3]);

        char fn[256];
        snprintf(fn, sizeof(fn), "/tmp/main24_weight_perturb_outputs/trial%d_mag%s.bin", t+1, mag_names[t]);
        FILE *f2 = fopen(fn, "wb"); fwrite(outp, 1, outBytes, f2); fclose(f2);

        // === Restore ===
        W0[patch_idx] = (_Float16)orig_f;
        NSData *wdataR = build_weight_blob(W0, CH, CH);
        [wdataR writeToFile:weightPath atomically:YES];
        ee = nil;
        ((BOOL(*)(id,SEL,unsigned int,NSError**))objc_msgSend)(mdl, @selector(unloadWithQoS:error:), 21, &ee);
        ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(mdl, @selector(loadWithQoS:options:error:), 21, @{}, &ee);
        req = mkreq();
        IOSurfaceLock(ioIn, 0, NULL);
        inp = (float*)IOSurfaceGetBaseAddress(ioIn);
        for (int c = 0; c < CH; c++)
            for (int s = 0; s < SP; s++)
                inp[c*SP+s] = (float)(c*SP + s + 1) * 0.01f;
        IOSurfaceUnlock(ioIn, 0, NULL);
        ee = nil;
        ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(mdl, @selector(evaluateWithQoS:options:request:error:), 21, @{}, req, &ee);
        float *outr = (float*)malloc(outBytes);
        IOSurfaceLock(ioOut, kIOSurfaceLockReadOnly, NULL);
        memcpy(outr, IOSurfaceGetBaseAddress(ioOut), outBytes);
        IOSurfaceUnlock(ioOut, kIOSurfaceLockReadOnly, NULL);

        float revert_diff = 0;
        for (int i = 0; i < CH*SP; i++) {
            float d = fabsf(outr[i] - baseline[i]);
            if (d > revert_diff) revert_diff = d;
        }
        int reverted_clean = (revert_diff < 1e-6f);
        printf("  revert_max_diff=%.6f clean=%d\n", revert_diff, reverted_clean);

        fprintf(jfp, "    {\"trial\": %d, \"perturbation\": %.3f, \"perturbed_value\": %.6f, "
                "\"max_diff\": %.6f, \"n_changed\": %d, \"n_nan\": %d, \"n_inf\": %d, "
                "\"output_changed\": %s, \"reverted_clean\": %s, "
                "\"unload_ms\": %.3f, \"load_ms\": %.3f, \"reload_total_ms\": %.3f, "
                "\"revert_max_diff\": %.6f}%s\n",
                t+1, mag, (float)perturbed, max_diff, n_changed, n_nan, n_inf,
                (n_changed > 0 ? "true" : "false"),
                (reverted_clean ? "true" : "false"),
                unload_ms, load_ms, reload_lat[t], revert_diff,
                (t<4?",":""));

        free(outp); free(outr);
    }

    // Reload latency stats
    double sum=0, mn=reload_lat[0], mx=reload_lat[0];
    for (int i = 0; i < 5; i++) { sum += reload_lat[i]; if (reload_lat[i]<mn)mn=reload_lat[i]; if (reload_lat[i]>mx)mx=reload_lat[i]; }
    double mean = sum/5.0;
    // simple median: sort copy
    double s[5]; memcpy(s, reload_lat, sizeof(s));
    for (int i = 0; i < 5; i++) for (int j = i+1; j < 5; j++) if (s[j]<s[i]) { double tmp=s[i]; s[i]=s[j]; s[j]=tmp; }
    double median = s[2];

    fprintf(jfp, "  ],\n  \"reload_latency\": {\"mean_ms\": %.3f, \"median_ms\": %.3f, \"min_ms\": %.3f, \"max_ms\": %.3f}\n}\n",
        mean, median, mn, mx);
    fclose(jfp);

    printf("\n=== Reload latency: mean=%.2fms median=%.2fms min=%.2fms max=%.2fms ===\n",
        mean, median, mn, mx);
    printf("\nResults written to /tmp/main24_weight_perturb_table.json\n");

    free(baseline); free(W0);
    CFRelease(ioIn); CFRelease(ioOut);
    return 0;
}}
