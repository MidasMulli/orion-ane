// test_weight_perturb_iosurface.m — Main 26 W1 Probe 2
// Pass perturbed weights via the weightsBuffer: IOSurface slot on
// _ANERequest requestWithInputs:...:weightsBuffer:perfStats:procedureIndex:
// without recompile/reload. Goal: cheap runtime weight injection.
//
// Method:
//   1. Compile + load model with W0 once.
//   2. Baseline eval with weightsBuffer=nil.
//   3. Build IOSurface, fill with perturbed W blob (a) full-128B-header form,
//      (b) raw fp16 payload only. Try both at +2.0 perturbation.
//   4. Eval with weightsBuffer set to that surface.
//   5. Compare against baseline. Also re-eval with nil afterwards to
//      check whether the model state was mutated.
//
// Synthetic conv kernel, no banking content.

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

static float* eval_once(id mdl, IOSurfaceRef ioIn, IOSurfaceRef ioOut, int CH, int SP,
                        int outBytes, Class g_AR, Class g_AIO, id wbuf_obj)
{
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
        @[wI], @[@0], @[wO], @[@0], wbuf_obj, nil, @0);

    NSError *ee = nil;
    BOOL ok = ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(mdl,
        @selector(evaluateWithQoS:options:request:error:), 21, @{}, req, &ee);
    if (!ok) {
        printf("    eval err: %s\n", ee?[[ee description] UTF8String]:"?");
        return NULL;
    }

    float *outp = (float*)malloc(outBytes);
    IOSurfaceLock(ioOut, kIOSurfaceLockReadOnly, NULL);
    memcpy(outp, IOSurfaceGetBaseAddress(ioOut), outBytes);
    IOSurfaceUnlock(ioOut, kIOSurfaceLockReadOnly, NULL);
    return outp;
}

static float diff_max(float *a, float *b, int n) {
    float m = 0;
    for (int i = 0; i < n; i++) {
        float d = fabsf(a[i] - b[i]);
        if (d > m) m = d;
    }
    return m;
}

static int diff_count(float *a, float *b, int n) {
    int c = 0;
    for (int i = 0; i < n; i++)
        if (fabsf(a[i] - b[i]) > 1e-6f) c++;
    return c;
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
    int inBytes = CH * SP * 4;

    _Float16 *W0 = (_Float16*)calloc(N, sizeof(_Float16));
    for (int i = 0; i < N; i++) {
        unsigned u = (i*2654435761u) ^ 0xdeadbeef;
        float f = ((u % 1000) / 1000.0f) - 0.5f;
        W0[i] = (_Float16)f;
    }
    int patch_idx = 0;
    float orig_f = (float)W0[patch_idx];
    NSString *mil = gen_mil(CH, SP);

    NSData *wdata0 = build_weight_blob(W0, CH, CH);
    NSDictionary *weights = @{@"@model_path/weights/weight.bin": @{@"offset":@0, @"data":wdata0}};
    NSData *milData = [mil dataUsingEncoding:NSUTF8StringEncoding];

    id desc = ((id(*)(Class,SEL,id,id,id))objc_msgSend)(g_D,
        @selector(modelWithMILText:weights:optionsPlist:), milData, weights, nil);
    id mdl = ((id(*)(Class,SEL,id))objc_msgSend)(g_I, @selector(inMemoryModelWithDescriptor:), desc);
    NSString *hx = ((id(*)(id,SEL))objc_msgSend)(mdl, @selector(hexStringIdentifier));
    NSString *td = [NSTemporaryDirectory() stringByAppendingPathComponent:hx];
    NSFileManager *fm = [NSFileManager defaultManager];
    [fm removeItemAtPath:td error:nil];
    [fm createDirectoryAtPath:[td stringByAppendingPathComponent:@"weights"]
        withIntermediateDirectories:YES attributes:nil error:nil];
    [milData writeToFile:[td stringByAppendingPathComponent:@"model.mil"] atomically:YES];
    [wdata0 writeToFile:[td stringByAppendingPathComponent:@"weights/weight.bin"] atomically:YES];

    NSError *e = nil;
    BOOL ok = ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(mdl,
        @selector(compileWithQoS:options:error:), 21, @{}, &e);
    if (!ok) { printf("FAIL compile: %s\n", e?[[e description] UTF8String]:"?"); return 1; }
    ok = ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(mdl,
        @selector(loadWithQoS:options:error:), 21, @{}, &e);
    if (!ok) { printf("FAIL load: %s\n", e?[[e description] UTF8String]:"?"); return 1; }
    printf("compiled+loaded hex=%s\n", [hx UTF8String]);

    IOSurfaceRef ioIn = make_surface(inBytes);
    IOSurfaceRef ioOut = make_surface(outBytes);

    // Baseline eval (weightsBuffer=nil)
    float *baseline = eval_once(mdl, ioIn, ioOut, CH, SP, outBytes, g_AR, g_AIO, nil);
    if (!baseline) { printf("baseline eval FAIL\n"); return 1; }
    printf("baseline[0..3]: [%.6f %.6f %.6f %.6f]\n",
        baseline[0], baseline[1], baseline[2], baseline[3]);

    // Build perturbed weight bufs
    _Float16 *Wp = (_Float16*)malloc(N * sizeof(_Float16));
    memcpy(Wp, W0, N * sizeof(_Float16));
    Wp[patch_idx] = (_Float16)(orig_f + 2.0f);

    NSData *wdataP_full = build_weight_blob(Wp, CH, CH); // 128B header + payload
    size_t raw_bytes = N * sizeof(_Float16);              // raw fp16 payload only

    // Variant A: 128B header + payload IOSurface
    IOSurfaceRef wsurfA = make_surface([wdataP_full length]);
    IOSurfaceLock(wsurfA, 0, NULL);
    memcpy(IOSurfaceGetBaseAddress(wsurfA), [wdataP_full bytes], [wdataP_full length]);
    IOSurfaceUnlock(wsurfA, 0, NULL);
    id wbufA = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_AIO, @selector(objectWithIOSurface:), wsurfA);

    // Variant B: raw fp16 payload only IOSurface
    IOSurfaceRef wsurfB = make_surface(raw_bytes);
    IOSurfaceLock(wsurfB, 0, NULL);
    memcpy(IOSurfaceGetBaseAddress(wsurfB), Wp, raw_bytes);
    IOSurfaceUnlock(wsurfB, 0, NULL);
    id wbufB = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_AIO, @selector(objectWithIOSurface:), wsurfB);

    FILE *jfp = fopen("/tmp/main26_w1/probe2_iosurface.json", "w");
    fprintf(jfp, "{\n  \"probe\": \"weightsBuffer_iosurface_runtime\",\n");
    fprintf(jfp, "  \"meta\": {\"perturbation\": 2.0, \"patch_idx\": %d, \"original_value\": %.6f},\n",
        patch_idx, orig_f);
    fprintf(jfp, "  \"variants\": [\n");

    // Variant A eval
    printf("\n=== Variant A: 128B header + payload (%lu bytes) ===\n", [wdataP_full length]);
    float *outA = eval_once(mdl, ioIn, ioOut, CH, SP, outBytes, g_AR, g_AIO, wbufA);
    if (outA) {
        float md = diff_max(outA, baseline, CH*SP);
        int   nc = diff_count(outA, baseline, CH*SP);
        printf("  max_diff=%.6f n_changed=%d\n", md, nc);
        printf("  out[0..3]: [%.6f %.6f %.6f %.6f]\n", outA[0], outA[1], outA[2], outA[3]);
        fprintf(jfp, "    {\"variant\": \"A_header_plus_payload\", \"bytes\": %lu, "
                "\"max_diff\": %.6f, \"n_changed\": %d, \"output_changed\": %s},\n",
                [wdataP_full length], md, nc, (nc>0?"true":"false"));
    } else {
        fprintf(jfp, "    {\"variant\": \"A_header_plus_payload\", \"error\": \"eval_failed\"},\n");
    }

    // Variant B eval
    printf("\n=== Variant B: raw fp16 payload (%zu bytes) ===\n", raw_bytes);
    float *outB = eval_once(mdl, ioIn, ioOut, CH, SP, outBytes, g_AR, g_AIO, wbufB);
    if (outB) {
        float md = diff_max(outB, baseline, CH*SP);
        int   nc = diff_count(outB, baseline, CH*SP);
        printf("  max_diff=%.6f n_changed=%d\n", md, nc);
        printf("  out[0..3]: [%.6f %.6f %.6f %.6f]\n", outB[0], outB[1], outB[2], outB[3]);
        fprintf(jfp, "    {\"variant\": \"B_raw_fp16_payload\", \"bytes\": %zu, "
                "\"max_diff\": %.6f, \"n_changed\": %d, \"output_changed\": %s},\n",
                raw_bytes, md, nc, (nc>0?"true":"false"));
    } else {
        fprintf(jfp, "    {\"variant\": \"B_raw_fp16_payload\", \"error\": \"eval_failed\"},\n");
    }

    // Followup: re-eval with nil weightsBuffer to check for state mutation
    printf("\n=== Followup: nil weightsBuffer after injection ===\n");
    float *outN = eval_once(mdl, ioIn, ioOut, CH, SP, outBytes, g_AR, g_AIO, nil);
    if (outN) {
        float md = diff_max(outN, baseline, CH*SP);
        int   nc = diff_count(outN, baseline, CH*SP);
        printf("  max_diff_vs_baseline=%.6f n_changed=%d\n", md, nc);
        fprintf(jfp, "    {\"variant\": \"C_post_nil_recheck\", "
                "\"max_diff_vs_baseline\": %.6f, \"n_changed\": %d, \"state_mutated\": %s}\n",
                md, nc, (nc>0?"true":"false"));
        free(outN);
    } else {
        fprintf(jfp, "    {\"variant\": \"C_post_nil_recheck\", \"error\": \"eval_failed\"}\n");
    }

    fprintf(jfp, "  ]\n}\n");
    fclose(jfp);

    if (outA) free(outA);
    if (outB) free(outB);
    free(baseline); free(Wp); free(W0);
    CFRelease(wsurfA); CFRelease(wsurfB);
    CFRelease(ioIn); CFRelease(ioOut);
    printf("\nResults written to /tmp/main26_w1/probe2_iosurface.json\n");
    return 0;
}}
