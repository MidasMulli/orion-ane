// test_native.m — Minimal native test to verify ANE in-memory model works
// Compile: clang -framework Foundation -framework IOSurface -framework CoreML -lobjc test_native.m -o test_native
#import <Foundation/Foundation.h>
#import <objc/runtime.h>
#import <objc/message.h>
#import <dlfcn.h>
#import <IOSurface/IOSurface.h>
#import <mach/mach_time.h>

int main() {
    @autoreleasepool {
        // Load private framework
        dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine", RTLD_NOW);

        Class Desc  = NSClassFromString(@"_ANEInMemoryModelDescriptor");
        Class IMM   = NSClassFromString(@"_ANEInMemoryModel");
        Class AR    = NSClassFromString(@"_ANERequest");
        Class AIO   = NSClassFromString(@"_ANEIOSurfaceObject");

        printf("Classes: Desc=%p IMM=%p AR=%p AIO=%p\n", Desc, IMM, AR, AIO);
        if (!Desc || !IMM || !AR || !AIO) {
            printf("FAIL: Missing classes\n");
            return 1;
        }

        // Try the exact MIL from gen_dynamic_matmul_mil(64, 64, 64)
        NSString *mil = @"program(1.3)\n"
            "[buildInfo = dict<string, string>({\"coremlc-component-MIL\", \"3510.2.1\"}, "
            "{\"coremlc-version\", \"3505.4.1\"}, {\"coremltools-component-milinternal\", \"\"}, "
            "{\"coremltools-version\", \"9.0\"})]\n"
            "{\n"
            "    func main<ios18>(tensor<fp32, [1, 64, 1, 128]> x) {\n"
            "        string to16 = const()[name = string(\"to16\"), val = string(\"fp16\")];\n"
            "        tensor<fp16, [1, 64, 1, 128]> xh = cast(dtype = to16, x = x)[name = string(\"cin\")];\n"
            "        tensor<int32, [4]> ba = const()[name = string(\"ba\"), val = tensor<int32, [4]>([0,0,0,0])];\n"
            "        tensor<int32, [4]> sa = const()[name = string(\"sa\"), val = tensor<int32, [4]>([1,64,1,64])];\n"
            "        tensor<fp16, [1,64,1,64]> act = slice_by_size(x=xh,begin=ba,size=sa)[name=string(\"act\")];\n"
            "        tensor<int32, [4]> bw = const()[name = string(\"bw\"), val = tensor<int32, [4]>([0,0,0,64])];\n"
            "        tensor<int32, [4]> sw = const()[name = string(\"sw\"), val = tensor<int32, [4]>([1,64,1,64])];\n"
            "        tensor<fp16, [1,64,1,64]> wt = slice_by_size(x=xh,begin=bw,size=sw)[name=string(\"wt\")];\n"
            "        tensor<int32, [4]> ra = const()[name = string(\"ra\"), val = tensor<int32, [4]>([1,1,64,64])];\n"
            "        tensor<fp16, [1,1,64,64]> a2 = reshape(shape=ra,x=act)[name=string(\"a2\")];\n"
            "        tensor<int32, [4]> pm = const()[name = string(\"pm\"), val = tensor<int32, [4]>([0,1,3,2])];\n"
            "        tensor<fp16, [1,1,64,64]> a3 = transpose(perm=pm,x=a2)[name=string(\"a3\")];\n"
            "        tensor<int32, [4]> rw = const()[name = string(\"rw\"), val = tensor<int32, [4]>([1,1,64,64])];\n"
            "        tensor<fp16, [1,1,64,64]> W = reshape(shape=rw,x=wt)[name=string(\"W\")];\n"
            "        bool bF = const()[name = string(\"bF\"), val = bool(false)];\n"
            "        tensor<fp16, [1,1,64,64]> yh = matmul(transpose_x=bF,transpose_y=bF,x=a3,y=W)[name=string(\"mm\")];\n"
            "        tensor<fp16, [1,1,64,64]> yt = transpose(perm=pm,x=yh)[name=string(\"yt\")];\n"
            "        tensor<int32, [4]> ro = const()[name = string(\"ro\"), val = tensor<int32, [4]>([1,64,1,64])];\n"
            "        tensor<fp16, [1,64,1,64]> yr = reshape(shape=ro,x=yt)[name=string(\"yr\")];\n"
            "        string to32 = const()[name = string(\"to32\"), val = string(\"fp32\")];\n"
            "        tensor<fp32, [1,64,1,64]> y = cast(dtype = to32, x = yr)[name = string(\"cout\")];\n"
            "    } -> (y);\n"
            "}\n";

        NSData *milData = [mil dataUsingEncoding:NSUTF8StringEncoding];
        printf("MIL text: %lu bytes\n", (unsigned long)milData.length);

        // Print first 200 chars of MIL for debug
        NSString *milPreview = [[NSString alloc] initWithData:[milData subdataWithRange:NSMakeRange(0, MIN(200, milData.length))] encoding:NSUTF8StringEncoding];
        printf("MIL preview: %s\n", [milPreview UTF8String]);

        // Create descriptor
        printf("Creating descriptor...\n");
        NSError *e = nil;
        id desc = ((id(*)(Class,SEL,id,id,id))objc_msgSend)(
            Desc, @selector(modelWithMILText:weights:optionsPlist:),
            milData, @{}, nil);
        printf("desc = %p\n", desc);
        if (!desc) {
            printf("FAIL: modelWithMILText returned nil\n");

            // Try with different function target versions
            printf("\nTrying ios17...\n");
            NSString *mil17 = [mil stringByReplacingOccurrencesOfString:@"ios18" withString:@"ios17"];
            NSData *md17 = [mil17 dataUsingEncoding:NSUTF8StringEncoding];
            desc = ((id(*)(Class,SEL,id,id,id))objc_msgSend)(
                Desc, @selector(modelWithMILText:weights:optionsPlist:),
                md17, @{}, nil);
            printf("ios17 desc = %p\n", desc);

            printf("\nTrying macos15...\n");
            NSString *mil15 = [mil stringByReplacingOccurrencesOfString:@"ios18" withString:@"macos15"];
            NSData *md15 = [mil15 dataUsingEncoding:NSUTF8StringEncoding];
            desc = ((id(*)(Class,SEL,id,id,id))objc_msgSend)(
                Desc, @selector(modelWithMILText:weights:optionsPlist:),
                md15, @{}, nil);
            printf("macos15 desc = %p\n", desc);

            printf("\nTrying macos16...\n");
            NSString *mil16 = [mil stringByReplacingOccurrencesOfString:@"ios18" withString:@"macos16"];
            NSData *md16 = [mil16 dataUsingEncoding:NSUTF8StringEncoding];
            desc = ((id(*)(Class,SEL,id,id,id))objc_msgSend)(
                Desc, @selector(modelWithMILText:weights:optionsPlist:),
                md16, @{}, nil);
            printf("macos16 desc = %p\n", desc);

            printf("\nTrying ios19...\n");
            NSString *mil19 = [mil stringByReplacingOccurrencesOfString:@"ios18" withString:@"ios19"];
            NSData *md19 = [mil19 dataUsingEncoding:NSUTF8StringEncoding];
            desc = ((id(*)(Class,SEL,id,id,id))objc_msgSend)(
                Desc, @selector(modelWithMILText:weights:optionsPlist:),
                md19, @{}, nil);
            printf("ios19 desc = %p\n", desc);

            // Try with no build info
            printf("\nTrying minimal MIL (no buildInfo)...\n");
            NSString *minMil = @"program(1.3)\n"
                "{\n"
                "    func main<ios18>(tensor<fp32, [1, 64, 1, 128]> x) {\n"
                "        string to16 = const()[name = string(\"to16\"), val = string(\"fp16\")];\n"
                "        tensor<fp16, [1, 64, 1, 128]> xh = cast(dtype = to16, x = x)[name = string(\"cin\")];\n"
                "        string to32 = const()[name = string(\"to32\"), val = string(\"fp32\")];\n"
                "        tensor<fp32, [1, 64, 1, 128]> y = cast(dtype = to32, x = xh)[name = string(\"cout\")];\n"
                "    } -> (y);\n"
                "}\n";
            NSData *minData = [minMil dataUsingEncoding:NSUTF8StringEncoding];
            desc = ((id(*)(Class,SEL,id,id,id))objc_msgSend)(
                Desc, @selector(modelWithMILText:weights:optionsPlist:),
                minData, @{}, nil);
            printf("minimal desc = %p\n", desc);

            if (!desc) {
                printf("\nAll MIL formats failed. This may be a macOS 26 compatibility issue.\n");

                // List methods on the descriptor class
                printf("\nMethods on %s:\n", class_getName(Desc));
                unsigned int count;
                Method *methods = class_copyMethodList(object_getClass(Desc), &count);
                for (unsigned int i = 0; i < count; i++) {
                    printf("  + %s\n", sel_getName(method_getName(methods[i])));
                }
                free(methods);
                methods = class_copyMethodList(Desc, &count);
                for (unsigned int i = 0; i < count; i++) {
                    printf("  - %s\n", sel_getName(method_getName(methods[i])));
                }
                free(methods);
                return 1;
            }
        }

        printf("\n✓ Descriptor created!\n");

        // Create model
        id model = ((id(*)(Class,SEL,id))objc_msgSend)(IMM, @selector(inMemoryModelWithDescriptor:), desc);
        printf("model = %p\n", model);

        if (model) {
            id hx = ((id(*)(id,SEL))objc_msgSend)(model, @selector(hexStringIdentifier));
            printf("hexId = %s\n", [hx UTF8String]);

            // Pre-populate temp dir
            NSString *td = [NSTemporaryDirectory() stringByAppendingPathComponent:hx];
            NSFileManager *fm = [NSFileManager defaultManager];
            [fm createDirectoryAtPath:[td stringByAppendingPathComponent:@"weights"]
                withIntermediateDirectories:YES attributes:nil error:nil];
            [milData writeToFile:[td stringByAppendingPathComponent:@"model.mil"] atomically:YES];

            // Compile
            printf("Compiling...\n");
            BOOL ok = ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(
                model, @selector(compileWithQoS:options:error:), 21, @{}, &e);
            printf("compile: %s\n", ok ? "YES" : "NO");
            if (e) printf("  err: %s\n", [[e description] UTF8String]);

            if (ok) {
                printf("Loading...\n");
                ok = ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(
                    model, @selector(loadWithQoS:options:error:), 21, @{}, &e);
                printf("load: %s\n", ok ? "YES" : "NO");
                if (e) printf("  err: %s\n", [[e description] UTF8String]);

                if (ok) {
                    int in_bytes = 64 * 128 * 4;
                    int out_bytes = 64 * 64 * 4;
                    IOSurfaceRef ioIn = IOSurfaceCreate((__bridge CFDictionaryRef)@{
                        (id)kIOSurfaceWidth:@(in_bytes),(id)kIOSurfaceHeight:@1,
                        (id)kIOSurfaceBytesPerElement:@1,(id)kIOSurfaceBytesPerRow:@(in_bytes),
                        (id)kIOSurfaceAllocSize:@(in_bytes),(id)kIOSurfacePixelFormat:@0});
                    IOSurfaceRef ioOut = IOSurfaceCreate((__bridge CFDictionaryRef)@{
                        (id)kIOSurfaceWidth:@(out_bytes),(id)kIOSurfaceHeight:@1,
                        (id)kIOSurfaceBytesPerElement:@1,(id)kIOSurfaceBytesPerRow:@(out_bytes),
                        (id)kIOSurfaceAllocSize:@(out_bytes),(id)kIOSurfacePixelFormat:@0});

                    // Fill input with identity test
                    IOSurfaceLock(ioIn, 0, NULL);
                    float *inp = (float*)IOSurfaceGetBaseAddress(ioIn);
                    for (int d = 0; d < 64; d++)
                        for (int s = 0; s < 64; s++)
                            inp[d * 128 + s] = (float)(d * 64 + s) * 0.001f;
                    for (int d = 0; d < 64; d++)
                        for (int c = 0; c < 64; c++)
                            inp[d * 128 + 64 + c] = (d == c) ? 1.0f : 0.0f;
                    IOSurfaceUnlock(ioIn, 0, NULL);

                    id wIn = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(AIO, @selector(objectWithIOSurface:), ioIn);
                    id wOut = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(AIO, @selector(objectWithIOSurface:), ioOut);
                    id req = ((id(*)(Class,SEL,id,id,id,id,id,id,id))objc_msgSend)(AR,
                        @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:),
                        @[wIn], @[@0], @[wOut], @[@0], nil, nil, @0);

                    printf("Evaluating...\n");
                    ok = ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(
                        model, @selector(evaluateWithQoS:options:request:error:),
                        21, @{}, req, &e);
                    printf("eval: %s\n", ok ? "YES" : "NO");
                    if (e) printf("  err: %s\n", [[e description] UTF8String]);

                    if (ok) {
                        IOSurfaceLock(ioOut, kIOSurfaceLockReadOnly, NULL);
                        float *out = (float*)IOSurfaceGetBaseAddress(ioOut);
                        printf("Output[0:8]: ");
                        for (int i = 0; i < 8; i++) printf("%.4f ", out[i]);
                        printf("\n");

                        // Benchmark
                        for (int i = 0; i < 10; i++)
                            ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(
                                model, @selector(evaluateWithQoS:options:request:error:),
                                21, @{}, req, &e);

                        mach_timebase_info_data_t tb;
                        mach_timebase_info(&tb);
                        int iters = 100;
                        uint64_t t0 = mach_absolute_time();
                        for (int i = 0; i < iters; i++)
                            ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(
                                model, @selector(evaluateWithQoS:options:request:error:),
                                21, @{}, req, &e);
                        double ms = (double)(mach_absolute_time() - t0) * tb.numer / tb.denom / 1e6 / iters;
                        double gflops = 2.0 * 64 * 64 * 64 / ms / 1e6;

                        printf("\n===== SUCCESS =====\n");
                        printf("64×64 matmul on ANE: %.3f ms/eval, %.2f GFLOPS\n", ms, gflops);
                        IOSurfaceUnlock(ioOut, kIOSurfaceLockReadOnly, NULL);
                    }

                    ((BOOL(*)(id,SEL,unsigned int,NSError**))objc_msgSend)(
                        model, @selector(unloadWithQoS:error:), 21, &e);
                    CFRelease(ioIn); CFRelease(ioOut);
                }
            }

            [fm removeItemAtPath:td error:nil];
        }
    }
    return 0;
}
