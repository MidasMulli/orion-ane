CC = xcrun clang
CFLAGS = -O2 -Wall -Wno-deprecated-declarations -fobjc-arc -fPIC
FRAMEWORKS = -framework Foundation -framework IOSurface -framework Accelerate -ldl
TARGET = libane_bridge.dylib

SOURCES = ane_bridge.m forward_pass.c
HEADERS = ane_bridge.h forward_pass.h

all: $(TARGET)

$(TARGET): $(SOURCES) $(HEADERS)
	$(CC) $(CFLAGS) -dynamiclib -o $@ $(SOURCES) $(FRAMEWORKS)

test: test_bridge.m ane_bridge.h $(TARGET)
	$(CC) $(CFLAGS) -o test_bridge test_bridge.m -L. -lane_bridge $(FRAMEWORKS)

clean:
	rm -f $(TARGET) test_bridge

.PHONY: all clean test
