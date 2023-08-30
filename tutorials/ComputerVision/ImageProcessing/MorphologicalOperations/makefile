# denoising_file="denoising.cpp"
denoising_file="denoising0.cpp"

# LDFLAGS="-Wl,--copy-dt-needed-entries"
# -Wl,--copy-dt-needed-entries
# -lboost_serialization

denoising:
	# g++ $(denoising_file) -o output -I/path/to/opencv/include -L/path/to/opencv/lib -lboost_system -lopencv_core -lopencv_highgui -lopencv_imgproc 
	g++ $(denoising_file) -o output -I/usr/local/include -L/path/to/opencv/lib -lopencv_core -lopencv_highgui -lopencv_imgproc 
    
