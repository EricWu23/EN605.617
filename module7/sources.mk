# Add your Source files to this variable
SOURCESLOCATION =	assignment.cu \
	$(COMMON)/utility.cu \
	$(PAGABLEMEM)/globalpagable.cu \
	$(SHAREDMEM)/sharedmem.cu \
	$(CONSTMEM)/constmem.cu \
	$(REGISTERS)/registermem.cu
# Add your Source files to this variable
SOURCES =	assignment.cu \
			utility.cu \
			globalpagable.cu \
			sharedmem.cu \
			constmem.cu \
			registermem.cu
# Add your files to be cleaned but not part of the project
IRRELEVANT =	register \
	module6_stretch_problem\
	module6_stretch_problem_debug
	
# Add your include paths to this variable
COMMON:=./common
CONSTMEM:=./constmem
PAGABLEMEM:=./pagablemem
REGISTERS:=./registers
SHAREDMEM:=./sharedmem

INCLUDES = $(COMMON) \
	$(CONSTMEM) \
	$(PAGABLEMEM) \
	$(REGISTERS) \
	$(SHAREDMEM)
