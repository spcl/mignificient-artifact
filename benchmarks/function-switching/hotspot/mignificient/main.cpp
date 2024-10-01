#include <string>
#include <iostream>
#include <chrono>

void run(
  int grid_rows, int pyramid_height, int total_iterations,
  const char* tfile, const char* pfile, const char* ofile
);

void usage(int argc, char **argv);

int main(int argc, char **argv)
{
  char *tfile, *pfile, *ofile;
  
  int total_iterations = 60;
  int pyramid_height = 1; // number of iterations
  int grid_rows, grid_cols;
                          //
	if (argc != 7)
		usage(argc, argv);
	if((grid_rows = atoi(argv[1]))<=0||
	   (grid_cols = atoi(argv[1]))<=0||
       (pyramid_height = atoi(argv[2]))<=0||
       (total_iterations = atoi(argv[3]))<=0)
		usage(argc, argv);
		
	tfile=argv[4];
    pfile=argv[5];
    ofile=argv[6];

  auto begin = std::chrono::high_resolution_clock::now();
  run(grid_rows, pyramid_height, total_iterations, tfile, pfile, ofile);
  auto end = std::chrono::high_resolution_clock::now();

  std::cerr << std::chrono::duration_cast<std::chrono::microseconds>(end-begin).count() / 1000.0 << std::endl;
}
