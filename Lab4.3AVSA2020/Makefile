# NOTE: The variables PATH_INCLUDES & PATH_LIB must match the location of your OpenCV installation  
#
# To set PATH_INCLUDE: 
#	To find the 'include' directory, type in a terminal:
#	$ find / -name opencv.hpp 2>&1 | grep -v "Permission denied"
#	$ /usr/local/include/opencv2/opencv.hpp
#	so your 'include' directory is located in '/usr/local/include/'
#
# To set PATH_LIB: 
#	To find the 'lib' directory, type in a terminal:
#	$ find / -name libopencv* 2>&1 | grep -v "Permission denied"
#	$ ...
#	$ /usr/local/lib/libopencv_videostab.so.3.0
#	$ ...
#	$ /usr/local/lib/libopencv_core.so.3.0.0
#	$ /usr/local/lib/libopencv_core.so.3.0
#	$ /usr/local/lib/libopencv_core.so
#	$ ...
#	so your 'lib' directory is located in '/usr/local/lib/'
#	and your OpenCV version is 3.0.0
#
#   In this sample code, OpenCV is installed in the path '/opt/installation'
#	with the following details:
#		Version: 3.4.4
#		Include Path: /opt/installation/OpenCV-3.4.4/include
#		Library Path: /opt/instllation/OpenCV-3.4.4/lib
#
#	Author: Juan C. SanMiguel (juancarlos.sanmiguel@uam.es)

LIBS = -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_videoio -lopencv_objdetect -lopencv_flann -lopencv_imgcodecs -lopencv_ml -lopencv_video -lopencv_features2d -lopencv_calib3d 
PATH_INCLUDES = /opt/installation/OpenCV-3.4.4/include
PATH_LIB = /opt/installation/OpenCV-3.4.4/lib

all: clean Lab4.3AVSA2020

Lab4.3AVSA2020: main.o utils.o ShowManyImages.o GradientBasedTracker.o
	g++ -o Lab4.3AVSA2020 main.o utils.o ShowManyImages.o GradientBasedTracker.o -L$(PATH_LIB) $(LIBS) -lm

main.o: src/main.cpp utils.o GradientBasedTracker.o
	g++ -c src/main.cpp -I$(PATH_INCLUDES) -O

utils.o: src/utils.cpp src/utils.hpp
	g++ -c src/utils.cpp -I$(PATH_INCLUDES) -O

GradientBasedTracker.o: src/GradientBasedTracker.cpp src/GradientBasedTracker.hpp
	g++ -c src/GradientBasedTracker.cpp -I$(PATH_INCLUDES) -O

ShowManyImages.o: src/ShowManyImages.cpp src/ShowManyImages.hpp
	g++ -c src/ShowManyImages.cpp -I$(PATH_INCLUDES) -O

clean:
	rm -f  *o Lab4.3AVSA2020
	

