CXX      := g++
NVCC     := nvcc
SRCDIR   := src
BLDDIR   := build
TARGET   := terrain

SRCS     := $(shell find $(SRCDIR) -name "*.cu")
OBJS_CPU := $(patsubst $(SRCDIR)/%.cu, $(BLDDIR)/cpu/%.o, $(SRCS))
OBJS_GPU := $(patsubst $(SRCDIR)/%.cu, $(BLDDIR)/gpu/%.o, $(SRCS))

CXXFLAGS  := -O2 -std=c++11 -I$(SRCDIR)
NVCCFLAGS := -O2 -arch=sm_86 -rdc=true -I$(SRCDIR)
NVCCFLAGS := -O2 -rdc=true -I$(SRCDIR) \
			 -gencode arch=compute_52,code=sm_52 \
			 -gencode arch=compute_61,code=sm_61 \
			 -gencode arch=compute_75,code=sm_75 \
			 -gencode arch=compute_86,code=sm_86 \
			 -gencode arch=compute_52,code=compute_52

.PHONY: cpu gpu clean

cpu: $(OBJS_CPU)
	$(CXX) $^ -o $(TARGET) -lm

$(BLDDIR)/cpu/%.o: $(SRCDIR)/%.cu
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -x c++ -c $< -o $@

gpu: $(OBJS_GPU)
	$(NVCC) $(NVCCFLAGS) $^ -o $(TARGET)

$(BLDDIR)/gpu/%.o: $(SRCDIR)/%.cu
	@mkdir -p $(dir $@)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

clean:
	rm -rf $(BLDDIR) $(TARGET)