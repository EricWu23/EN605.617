# Add your Source files to this variable
SOURCESLOCATION =	assignment.cu \
	$(NVGRAPHSSP)/nvgraph_SSSP.cu

# Add your Source files to this variable
SOURCES =	assignment.cu \
			nvgraph_SSSP.cu

# Add your files to be cleaned but not part of the project
IRRELEVANT =

	
# Add your include paths to this variable
NVGRAPHSSP:=./nvgraph_SSSP
EXTERNAL=../common/inc

INCLUDES = $(NVGRAPHSSP) \
	$(EXTERNAL)


	
LIBRARIESNVGRAPH :=
LIBRARIESNVGRAPH += -lnvgraph