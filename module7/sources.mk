# Add your Source files to this variable
SOURCESLOCATION =	assignment.cu \
	$(COMMON)/utility.cu \
	$(PAGABLEMEM)/globalpagable.cu \
	$(SHAREDMEM)/sharedmem.cu \
	$(CONSTMEM)/constmem.cu \
	$(REGISTERS)/registermem.cu \
	$(PINNEDMEM)/globalpinnedmem.cu \
	$(STREAM)/stream.cu
# Add your Source files to this variable
SOURCES =	assignment.cu \
			utility.cu \
			globalpagable.cu \
			sharedmem.cu \
			constmem.cu \
			registermem.cu \
			globalpinnedmem.cu \
			stream.cu
# Add your files to be cleaned but not part of the project
IRRELEVANT =	multi_gpu_example \
				multiGPU \
				stream_example

	
# Add your include paths to this variable
COMMON:=./common
CONSTMEM:=./constmem
PAGABLEMEM:=./pagablemem
REGISTERS:=./registers
SHAREDMEM:=./sharedmem
PINNEDMEM:=./pinnedmem
STREAM:=./streamevent

INCLUDES = $(COMMON) \
	$(CONSTMEM) \
	$(PAGABLEMEM) \
	$(REGISTERS) \
	$(SHAREDMEM) \
	$(PINNEDMEM) \
	$(STREAM)
