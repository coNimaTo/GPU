CXX      := g++
NVCC     := nvcc
SRCDIR   := src
BLDDIR   := build
TARGET   := terrain

SRCS     := $(shell find $(SRCDIR) -name "*.cu")
OBJS_CPU := $(patsubst $(SRCDIR)/%.cu, $(BLDDIR)/cpu/%.o, $(SRCS))
OBJS_GPU := $(patsubst $(SRCDIR)/%.cu, $(BLDDIR)/gpu/%.o, $(SRCS))

CXXFLAGS  := -O2 -std=c++11 -I$(SRCDIR)
NVCCFLAGS := -O2 -arch=sm_75 -rdc=true -I$(SRCDIR)

.PHONY: cpu gpu clean

cpu: $(OBJS_CPU)
	$(CXX) $^ -o $(TARGET) -lm

$(BLDDIR)/cpu/%.o: $(SRCDIR)/%.cu
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -x c++ -c $< -o $@

gpu: $(OBJS_GPU)
	$(NVCC) $^ -o $(TARGET)

$(BLDDIR)/gpu/%.o: $(SRCDIR)/%.cu
	@mkdir -p $(dir $@)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

clean:
	rm -rf $(BLDDIR) $(TARGET)