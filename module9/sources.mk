# Add your Source files to this variable
SOURCESLOCATION =	assignment.cu \
	$(COMMON)/utility.cu \
	$(PAGABLEMEM)/globalpagable.cu \
	$(PINNEDMEM)/globalpinnedmem.cu \
	$(THRUST)/thrustmath.cu
# Add your Source files to this variable
SOURCES =	assignment.cu \
			utility.cu \
			globalpagable.cu \
			globalpinnedmem.cu \
			thrustmath.cu
# Add your files to be cleaned but not part of the project
IRRELEVANT =

	
# Add your include paths to this variable
COMMON:=./common
PAGABLEMEM:=./pagablemem
PINNEDMEM:=./pinnedmem
THRUST:=./thrust

INCLUDES = $(COMMON) \
	$(PAGABLEMEM) \
	$(PINNEDMEM) \
	$(THRUST)

