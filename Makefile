src=src/c_api.cpp src/gbm.cpp src/tree.cpp src/utils.cpp
main:$(src)
	g++ -fPIC -fopenmp -shared -o lib_xgbm.so $(src)

