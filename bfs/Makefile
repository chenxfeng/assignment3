all: default grade

default: main.cpp bfs.cpp
	g++ -std=c++11 -fopenmp -O3 -g -o bfs main.cpp bfs.cpp ../common/graph.cpp
grade: grade.cpp bfs.cpp
	g++ -std=c++11 -fopenmp -O3 -g -o bfs_grader grade.cpp bfs.cpp ../common/graph.cpp
clean:
	rm -rf bfs_grader bfs  *~ *.*~
