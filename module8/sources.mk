# Add your Source files to this variable
SOURCESLOCATION =	assignment.cu \
	$(COMMON)/utility.cu \
	$(CUBLAS)/matrixmultiplication.cu \
	$(CUBLAS)/unitTest_matrixmultiply.cu \
	$(CUSOLVER)/linearsolve.cu \
	$(CUSOLVER)/unitTest_linearsolve.cu
# Add your Source files to this variable
SOURCES =	assignment.cu \
			utility.cu \
			matrixmultiplication.cu \
			unitTest_matrixmultiply.cu \
			linearsolve.cu \
			unitTest_linearsolve.cu
# Add your files to be cleaned but not part of the project
IRRELEVANT =	cublas_example \
				cufft_example \
				curand_example \
				cusolver_example

	
# Add your include paths to this variable
COMMON:=./common
CUBLAS:=./cublas
CUSOLVER:=./cusolver

INCLUDES = $(COMMON) \
	$(CUBLAS) \
	$(CUSOLVER)
