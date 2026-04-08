// test_coreml_mil.m — Test with EXACT CoreML-generated MIL format
#import <Foundation/Foundation.h>
#import <objc/runtime.h>
#import <objc/message.h>
#import <dlfcn.h>
#import <IOSurface/IOSurface.h>
#import <mach/mach_time.h>

int main() {
    @autoreleasepool {
        dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine", RTLD_NOW);

        Class Desc = NSClassFromString(@"_ANEInMemoryModelDescriptor");
        Class IMM  = NSClassFromString(@"_ANEInMemoryModel");
        Class AR   = NSClassFromString(@"_ANERequest");
        Class AIO  = NSClassFromString(@"_ANEIOSurfaceObject");

        // Read EXACT MIL from CoreML compiler output
        NSString *milPath = @"/tmp/ane_simple.mlmodelc/model.mil";
        NSData *milData = [NSData dataWithContentsOfFile:milPath];
        if (!milData) {
            printf("FAIL: Can't read %s\n", [milPath UTF8String]);
            return 1;
        }
        printf("Read MIL: %lu bytes from %s\n", (unsigned long)milData.length, [milPath UTF8String]);

        // Create descriptor
        NSError *e = nil;
        id desc = ((id(*)(Class,SEL,id,id,id))objc_msgSend)(
            Desc, @selector(modelWithMILText:weights:optionsPlist:),
            milData, @{}, nil);
        printf("desc = %p\n", desc);
        if (!desc) { printf("FAIL: desc nil\n"); return 1; }

        id model = ((id(*)(Class,SEL,id))objc_msgSend)(IMM, @selector(inMemoryModelWithDescriptor:), desc);
        printf("model = %p\n", model);

        id hx = ((id(*)(id,SEL))objc_msgSend)(model, @selector(hexStringIdentifier));
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
        if (e) { printf("  err: %s\n", [[e description] UTF8String]); e = nil; }

        if (ok) {
            printf("Loading...\n");
            ok = ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(
                model, @selector(loadWithQoS:options:error:), 21, @{}, &e);
            printf("load: %s\n", ok ? "YES" : "NO");
            if (e) { printf("  err: %s\n", [[e description] UTF8String]); e = nil; }

            if (ok) {
                int bytes = 64 * 64 * 4;
                IOSurfaceRef ioIn = IOSurfaceCreate((__bridge CFDictionaryRef)@{
                    (id)kIOSurfaceWidth:@(bytes),(id)kIOSurfaceHeight:@1,
                    (id)kIOSurfaceBytesPerElement:@1,(id)kIOSurfaceBytesPerRow:@(bytes),
                    (id)kIOSurfaceAllocSize:@(bytes),(id)kIOSurfacePixelFormat:@0});
                IOSurfaceRef ioOut = IOSurfaceCreate((__bridge CFDictionaryRef)@{
                    (id)kIOSurfaceWidth:@(bytes),(id)kIOSurfaceHeight:@1,
                    (id)kIOSurfaceBytesPerElement:@1,(id)kIOSurfaceBytesPerRow:@(bytes),
                    (id)kIOSurfaceAllocSize:@(bytes),(id)kIOSurfacePixelFormat:@0});

                IOSurfaceLock(ioIn, 0, NULL);
                float *inp = (float*)IOSurfaceGetBaseAddress(ioIn);
                for (int i = 0; i < 64*64; i++) inp[i] = (float)i * 0.001f;
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
                if (e) { printf("  err: %s\n", [[e description] UTF8String]); e = nil; }

                if (ok) {
                    IOSurfaceLock(ioOut, kIOSurfaceLockReadOnly, NULL);
                    float *out = (float*)IOSurfaceGetBaseAddress(ioOut);
                    printf("\nInput[0:4]:  %.4f %.4f %.4f %.4f\n", inp[0], inp[1], inp[2], inp[3]);
                    printf("Output[0:4]: %.4f %.4f %.4f %.4f\n", out[0], out[1], out[2], out[3]);
                    printf("Expected:    %.4f %.4f %.4f %.4f  (x+x)\n", inp[0]*2, inp[1]*2, inp[2]*2, inp[3]*2);

                    // Benchmark
                    for (int i = 0; i < 10; i++)
                        ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(
                            model, @selector(evaluateWithQoS:options:request:error:), 21, @{}, req, &e);

                    mach_timebase_info_data_t tb;
                    mach_timebase_info(&tb);
                    int iters = 1000;
                    uint64_t t0 = mach_absolute_time();
                    for (int i = 0; i < iters; i++)
                        ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(
                            model, @selector(evaluateWithQoS:options:request:error:), 21, @{}, req, &e);
                    double ms = (double)(mach_absolute_time() - t0) * tb.numer / tb.denom / 1e6 / iters;

                    printf("\n========================================\n");
                    printf("ANE IN-MEMORY EXECUTION: SUCCESS!\n");
                    printf("  %.4f ms per eval (%d iters)\n", ms, iters);
                    printf("  Tensor: [1, 64, 1, 64] = 4096 elements\n");
                    printf("  Op: element-wise add (x + x)\n");
                    printf("========================================\n");
                    IOSurfaceUnlock(ioOut, kIOSurfaceLockReadOnly, NULL);
                }

                ((BOOL(*)(id,SEL,unsigned int,NSError**))objc_msgSend)(
                    model, @selector(unloadWithQoS:error:), 21, &e);
                CFRelease(ioIn); CFRelease(ioOut);
            }
        }

        [fm removeItemAtPath:td error:nil];
    }
    return 0;
}
