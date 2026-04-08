// test_weight_perturb_full_compile.m — Main 26 W1 Probe 1
// FULL recompile per trial. Builds a fresh _ANEInMemoryModel object,
// fresh temp dir, fresh compileWithQoS:, fresh loadWithQoS: for each
// perturbation. The hypothesis under test: does the espresso/aned
// compile path actually re-ingest BLOBFILE weights when called fresh?
//
// Forked from test_weight_perturb_sweep.m. Synthetic conv kernel,
// no banking content.
//
// Single perturbation magnitudes: [+0.5, +2.0, +10.0]. We compile a
// brand-new model object per trial AND clear the per-model temp dir
// before each compile, eliminating any persistent compiled-state cache
// keyed on hexStringIdentifier (which is content-derived).

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

// Build, compile, load, eval one fresh model with the given weight blob.
// Returns malloc'd output buffer (caller frees) and fills *out_compile_ms,
// *out_load_ms, *out_hex.
static float* compile_load_eval(int CH, int SP, _Float16 *W, int N,
                                NSString *milString,
                                IOSurfaceRef ioIn, IOSurfaceRef ioOut, int outBytes,
                                Class g_D, Class g_I, Class g_AR, Class g_AIO,
                                double *out_compile_ms, double *out_load_ms,
                                NSString **out_hex)
{
    NSData *wdata = build_weight_blob(W, CH, CH);
    NSDictionary *weights = @{@"@model_path/weights/weight.bin": @{@"offset":@0, @"data":wdata}};
    NSData *milData = [milString dataUsingEncoding:NSUTF8StringEncoding];

    id desc = ((id(*)(Class,SEL,id,id,id))objc_msgSend)(g_D,
        @selector(modelWithMILText:weights:optionsPlist:), milData, weights, nil);
    id mdl = ((id(*)(Class,SEL,id))objc_msgSend)(g_I,
        @selector(inMemoryModelWithDescriptor:), desc);
    NSString *hx = ((id(*)(id,SEL))objc_msgSend)(mdl, @selector(hexStringIdentifier));
    if (out_hex) *out_hex = hx;

    NSString *td = [NSTemporaryDirectory() stringByAppendingPathComponent:hx];
    NSFileManager *fm = [NSFileManager defaultManager];
    // Aggressively wipe any prior cached state for this hex
    [fm removeItemAtPath:td error:nil];
    [fm createDirectoryAtPath:[td stringByAppendingPathComponent:@"weights"]
        withIntermediateDirectories:YES attributes:nil error:nil];
    [milData writeToFile:[td stringByAppendingPathComponent:@"model.mil"] atomically:YES];
    [wdata writeToFile:[td stringByAppendingPathComponent:@"weights/weight.bin"] atomically:YES];

    NSError *e = nil;
    uint64_t tc0 = mach_absolute_time();
    BOOL ok = ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(mdl,
        @selector(compileWithQoS:options:error:), 21, @{}, &e);
    uint64_t tc1 = mach_absolute_time();
    if (!ok) { printf("    FAIL compile: %s\n", e?[[e description] UTF8String]:"?"); return NULL; }
    ok = ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(mdl,
        @selector(loadWithQoS:options:error:), 21, @{}, &e);
    uint64_t tc2 = mach_absolute_time();
    if (!ok) { printf("    FAIL load: %s\n", e?[[e description] UTF8String]:"?"); return NULL; }
    if (out_compile_ms) *out_compile_ms = tb_ms(tc1 - tc0);
    if (out_load_ms)    *out_load_ms    = tb_ms(tc2 - tc1);

    // Re-write input deterministically
    IOSurfaceLock(ioIn, 0, NULL);
    float *inp = (float*)IOSurfaceGetBaseAddress(ioIn);
    for (int c = 0; c < CH; c++)
        for (int s = 0; s < SP; s++)
            inp[c*SP+s] = (float)(c*SP + s + 1) * 0.01f;
    IOSurfaceUnlock(ioIn, 0, NULL);

    id wI = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_AIO, @selector(objectWithIOSurface:), ioIn);
    id wO = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_AIO, @selector(objectWithIOSurface:), ioOut);
    id req = ((id(*)(Class,SEL,id,id,id,id,id,id,id))objc_msgSend)(g_AR,
        @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:),
        @[wI], @[@0], @[wO], @[@0], nil, nil, @0);

    NSError *ee = nil;
    ok = ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(mdl,
        @selector(evaluateWithQoS:options:request:error:), 21, @{}, req, &ee);
    if (!ok) { printf("    FAIL eval: %s\n", ee?[[ee description] UTF8String]:"?"); return NULL; }

    float *outp = (float*)malloc(outBytes);
    IOSurfaceLock(ioOut, kIOSurfaceLockReadOnly, NULL);
    memcpy(outp, IOSurfaceGetBaseAddress(ioOut), outBytes);
    IOSurfaceUnlock(ioOut, kIOSurfaceLockReadOnly, NULL);

    // Try to unload to release the kext-side state cleanly
    ee = nil;
    ((BOOL(*)(id,SEL,unsigned int,NSError**))objc_msgSend)(mdl,
        @selector(unloadWithQoS:error:), 21, &ee);

    return outp;
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
    int outBytes = CH * SP * 4;

    _Float16 *W0 = (_Float16*)calloc(N, sizeof(_Float16));
    for (int i = 0; i < N; i++) {
        unsigned u = (i*2654435761u) ^ 0xdeadbeef;
        float f = ((u % 1000) / 1000.0f) - 0.5f;
        W0[i] = (_Float16)f;
    }
    int patch_idx = 0;
    float orig_f = (float)W0[patch_idx];
    NSString *mil = gen_mil(CH, SP);

    int inBytes = CH * SP * 4;
    IOSurfaceRef ioIn = make_surface(inBytes);
    IOSurfaceRef ioOut = make_surface(outBytes);

    // === Baseline: fresh compile with W0 ===
    printf("=== BASELINE ===\n");
    double bc_ms=0, bl_ms=0;
    NSString *bhex = nil;
    float *baseline = compile_load_eval(CH, SP, W0, N, mil, ioIn, ioOut, outBytes,
        g_D, g_I, g_AR, g_AIO, &bc_ms, &bl_ms, &bhex);
    if (!baseline) { printf("BASELINE FAIL\n"); return 1; }
    printf("baseline compile=%.1fms load=%.1fms hex=%s\n", bc_ms, bl_ms, [bhex UTF8String]);
    printf("baseline[0..3]: [%.6f %.6f %.6f %.6f]\n",
        baseline[0], baseline[1], baseline[2], baseline[3]);

    FILE *fp = fopen("/tmp/main26_w1/baseline_p1.bin", "wb");
    if (fp) { fwrite(baseline, 1, outBytes, fp); fclose(fp); }

    float mags[3] = {0.5f, 2.0f, 10.0f};
    const char *mag_names[3] = {"+0.5","+2.0","+10.0"};

    FILE *jfp = fopen("/tmp/main26_w1/probe1_full_compile.json", "w");
    fprintf(jfp, "{\n  \"probe\": \"full_compile_per_trial\",\n");
    fprintf(jfp, "  \"meta\": {\n");
    fprintf(jfp, "    \"kernel\": \"conv %dx%d sp=%d, fp16, BLOBFILE weights\",\n", CH, CH, SP);
    fprintf(jfp, "    \"patch_idx\": %d, \"original_value\": %.6f,\n", patch_idx, orig_f);
    fprintf(jfp, "    \"baseline_compile_ms\": %.3f, \"baseline_load_ms\": %.3f,\n", bc_ms, bl_ms);
    fprintf(jfp, "    \"baseline_hex\": \"%s\",\n", [bhex UTF8String]);
    fprintf(jfp, "    \"note\": \"fresh _ANEInMemoryModel + wiped tmpdir + compileWithQoS+loadWithQoS per trial\"\n");
    fprintf(jfp, "  },\n  \"trials\": [\n");

    for (int t = 0; t < 3; t++) {
        float mag = mags[t];
        _Float16 perturbed = (_Float16)(orig_f + mag);
        printf("\n=== Trial %d: %s (perturbed=%.6f) ===\n", t+1, mag_names[t], (float)perturbed);

        _Float16 *Wp = (_Float16*)malloc(N * sizeof(_Float16));
        memcpy(Wp, W0, N * sizeof(_Float16));
        Wp[patch_idx] = perturbed;

        double cc_ms=0, ll_ms=0;
        NSString *phex = nil;
        float *outp = compile_load_eval(CH, SP, Wp, N, mil, ioIn, ioOut, outBytes,
            g_D, g_I, g_AR, g_AIO, &cc_ms, &ll_ms, &phex);
        if (!outp) {
            fprintf(jfp, "    {\"trial\": %d, \"error\": \"compile_load_eval failed\"}%s\n",
                t+1, (t<2?",":""));
            free(Wp);
            continue;
        }
        printf("  compile=%.1fms load=%.1fms hex=%s\n", cc_ms, ll_ms, [phex UTF8String]);
        printf("  out[0..3]: [%.6f %.6f %.6f %.6f]\n", outp[0], outp[1], outp[2], outp[3]);

        float max_diff = 0; int n_changed = 0; int n_nan = 0; int n_inf = 0;
        for (int i = 0; i < CH*SP; i++) {
            float d = fabsf(outp[i] - baseline[i]);
            if (isnan(outp[i])) n_nan++;
            if (isinf(outp[i])) n_inf++;
            if (d > max_diff) max_diff = d;
            if (d > 1e-6f) n_changed++;
        }
        int hex_changed = ![phex isEqualToString:bhex];
        printf("  max_diff=%.6f n_changed=%d/%d hex_changed=%d\n",
            max_diff, n_changed, CH*SP, hex_changed);

        char fn[256];
        snprintf(fn, sizeof(fn), "/tmp/main26_w1/probe1_trial%d_mag%s.bin", t+1, mag_names[t]);
        FILE *f2 = fopen(fn, "wb"); if (f2) { fwrite(outp, 1, outBytes, f2); fclose(f2); }

        fprintf(jfp, "    {\"trial\": %d, \"perturbation\": %.3f, \"perturbed_value\": %.6f, "
                "\"hex\": \"%s\", \"hex_changed_vs_baseline\": %s, "
                "\"compile_ms\": %.3f, \"load_ms\": %.3f, "
                "\"max_diff\": %.6f, \"n_changed\": %d, \"n_nan\": %d, \"n_inf\": %d, "
                "\"output_changed\": %s}%s\n",
                t+1, mag, (float)perturbed, [phex UTF8String],
                (hex_changed?"true":"false"), cc_ms, ll_ms,
                max_diff, n_changed, n_nan, n_inf,
                (n_changed > 0 ? "true" : "false"),
                (t<2?",":""));

        free(outp); free(Wp);
    }

    fprintf(jfp, "  ]\n}\n");
    fclose(jfp);

    printf("\nResults written to /tmp/main26_w1/probe1_full_compile.json\n");

    free(baseline); free(W0);
    CFRelease(ioIn); CFRelease(ioOut);
    return 0;
}}
