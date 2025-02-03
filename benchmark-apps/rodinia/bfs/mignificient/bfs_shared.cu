
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <cuda.h>


#include <chrono>
#include <vector>
#include <iostream>

#define MAX_THREADS_PER_BLOCK 512

int no_of_nodes;
int edge_list_size;
FILE *fp;

//Structure to hold a node information
struct Node
{
	int starting;
	int no_of_edges;
};

#include "kernel.cu"
#include "kernel2.cu"

void BFSGraph(const std::string & file);

////////////////////////////////////////////////////////////////////////////////
// Main Program
////////////////////////////////////////////////////////////////////////////////

#include "function.hpp"

extern "C" size_t function()
{

        no_of_nodes=0;
        edge_list_size=0;
        BFSGraph("graph1MW_6.txt");

        return 0;
}

////////////////////////////////////////////////////////////////////////////////
//Apply BFS on a Graph using CUDA
////////////////////////////////////////////////////////////////////////////////
void BFSGraph(const std::string & file) 
{
	//Read in Graph from a file
	fp = fopen(file.c_str(),"r");
	if(!fp)
	{
		printf("Error Reading graph file\n");
		return;
	}

	int source = 0;

	fscanf(fp,"%d",&no_of_nodes);

	int num_of_blocks = 1;
	int num_of_threads_per_block = no_of_nodes;

	//Make execution Parameters according to the number of nodes
	//Distribute threads across multiple Blocks if necessary
	if(no_of_nodes>MAX_THREADS_PER_BLOCK)
	{
		num_of_blocks = (int)ceil(no_of_nodes/(double)MAX_THREADS_PER_BLOCK); 
		num_of_threads_per_block = MAX_THREADS_PER_BLOCK; 
	}

	// allocate host memory
	Node* h_graph_nodes = (Node*) malloc(sizeof(Node)*no_of_nodes);
	bool *h_graph_mask = (bool*) malloc(sizeof(bool)*no_of_nodes);
	bool *h_updating_graph_mask = (bool*) malloc(sizeof(bool)*no_of_nodes);
	bool *h_graph_visited = (bool*) malloc(sizeof(bool)*no_of_nodes);
	int* h_cost = (int*) malloc( sizeof(int)*no_of_nodes);

	int start, edgeno;   
	// initalize the memory
	for( unsigned int i = 0; i < no_of_nodes; i++) 
	{
		fscanf(fp,"%d %d",&start,&edgeno);
		h_graph_nodes[i].starting = start;
		h_graph_nodes[i].no_of_edges = edgeno;
		h_graph_mask[i]=false;
		h_updating_graph_mask[i]=false;
		h_graph_visited[i]=false;
	}

	//read the source node from the file
	fscanf(fp,"%d",&source);
	source=0;

	//set the source node as true in the mask
	h_graph_mask[source]=true;
	h_graph_visited[source]=true;

	fscanf(fp,"%d",&edge_list_size);

	int id,cost;
	//int* h_graph_edges = (int*) malloc(sizeof(int)*edge_list_size);
	int* h_graph_edges = (int*) mignificient_malloc(sizeof(int)*edge_list_size);
	for(int i=0; i < edge_list_size ; i++)
	{
		fscanf(fp,"%d",&id);
		fscanf(fp,"%d",&cost);
		h_graph_edges[i] = id;
	}

	if(fp)
		fclose(fp);    

	//printf("Read File\n");

	//Copy the Node list to device memory
    	//auto s = std::chrono::high_resolution_clock::now();

	Node* d_graph_nodes;
	//printf("Malloc nodes");
	//cudaMalloc( (void**) &d_graph_nodes, sizeof(Node)*no_of_nodes) ;
	int* d_graph_edges;
	//printf("Malloc edges");
	cudaMalloc( (void**) &d_graph_edges, sizeof(int)*edge_list_size) ;
	bool* d_graph_mask;
	//printf("Malloc mask");
	cudaMalloc( (void**) &d_graph_mask, sizeof(bool)*no_of_nodes) ;
	bool* d_graph_visited;
	//printf("Malloc visited");
	cudaMalloc( (void**) &d_graph_visited, sizeof(bool)*no_of_nodes) ;
	bool* d_updating_graph_mask;
	//printf("Malloc updating graph mask");
	cudaMalloc( (void**) &d_updating_graph_mask, sizeof(bool)*no_of_nodes) ;

	// allocate device memory for result
	int* d_cost;
	//printf("Malloc cost");
	cudaMalloc( (void**) &d_cost, sizeof(int)*no_of_nodes);

	//make a bool to check if the execution is over
	bool *d_over;
	//printf("Malloc over");
	cudaMalloc( (void**) &d_over, sizeof(bool));

    	//auto sz = std::chrono::high_resolution_clock::now();
	//printf("Memcpy graph nodes");
	cudaMemcpy( d_graph_nodes, h_graph_nodes, sizeof(Node)*no_of_nodes, cudaMemcpyHostToDevice) ;

	//Copy the Edge List to device Memory
	//printf("Memcpy graph edges");
	cudaMemcpy( d_graph_edges, h_graph_edges, sizeof(int)*edge_list_size, cudaMemcpyHostToDevice) ;

	//Copy the Mask to device memory
	//printf("Memcpy graph mask");
	cudaMemcpy( d_graph_mask, h_graph_mask, sizeof(bool)*no_of_nodes, cudaMemcpyHostToDevice) ;

	//printf("Memcpy graph updating mask");
	cudaMemcpy( d_updating_graph_mask, h_updating_graph_mask, sizeof(bool)*no_of_nodes, cudaMemcpyHostToDevice) ;

	//Copy the Visited nodes array to device memory
	//printf("Memcpy graph visited");
	cudaMemcpy( d_graph_visited, h_graph_visited, sizeof(bool)*no_of_nodes, cudaMemcpyHostToDevice) ;


	// allocate mem for the result on host side
	for(int i=0;i<no_of_nodes;i++)
		h_cost[i]=-1;
	h_cost[source]=0;
	
	//printf("Memcpy graph cost");
	cudaMemcpy( d_cost, h_cost, sizeof(int)*no_of_nodes, cudaMemcpyHostToDevice) ;
//		cudaDeviceSynchronize();		
//    auto ez = std::chrono::high_resolution_clock::now();
//    auto dz = std::chrono::duration_cast<std::chrono::microseconds>(ez-sz).count() / 1000000.0;

	// setup execution parameters
	dim3  grid( num_of_blocks, 1, 1);
	dim3  threads( num_of_threads_per_block, 1, 1);

	int k=0;
	//printf("Start traversing the tree\n");
    //auto s1 = std::chrono::high_resolution_clock::now();
	bool stop;
	//Call the Kernel untill all the elements of Frontier are not false
	double time_kernel = 0.0;
	double time_copy = 0.0;
	do
	{
		//if no thread changes this value then the loop stops
		stop=false;
    //auto s_c = std::chrono::high_resolution_clock::now();
		cudaMemcpy( d_over, &stop, sizeof(bool), cudaMemcpyHostToDevice) ;
    //auto s_k = std::chrono::high_resolution_clock::now();
		Kernel<<< grid, threads, 0 >>>( d_graph_nodes, d_graph_edges, d_graph_mask, d_updating_graph_mask, d_graph_visited, d_cost, no_of_nodes);
		// check if kernel execution generated and error
		

		Kernel2<<< grid, threads, 0 >>>( d_graph_mask, d_updating_graph_mask, d_graph_visited, d_over, no_of_nodes);
		// check if kernel execution generated and error

		//cudaDeviceSynchronize();		

    //auto s_k2 = std::chrono::high_resolution_clock::now();
		cudaMemcpy( &stop, d_over, sizeof(bool), cudaMemcpyDeviceToHost) ;
    //auto e_c2  = std::chrono::high_resolution_clock::now();
    //time_kernel += std::chrono::duration_cast<std::chrono::nanoseconds>(s_k2-s_k).count() / 1000.0;
    //time_copy += std::chrono::duration_cast<std::chrono::nanoseconds>(e_c2-s_k2).count() / 1000.0;
    //time_copy += std::chrono::duration_cast<std::chrono::nanoseconds>(s_k-s_c).count() / 1000.0;
		k++;
	}
	while(stop);



	// copy result from device to host
	//printf("Memcpy back cost");
	cudaMemcpy( h_cost, d_cost, sizeof(int)*no_of_nodes, cudaMemcpyDeviceToHost) ;
    //auto e = std::chrono::high_resolution_clock::now();
    //auto d = std::chrono::duration_cast<std::chrono::microseconds>(e-s).count() / 1000000.0;

    //    printf("Kernel Executed %d times\n",k);
    //auto dz2 = std::chrono::duration_cast<std::chrono::microseconds>(sz-s).count() / 1000000.0;
    //    printf("Memory done %.8f\n\n", dz2);
    //    printf("Copied Everything to GPU memory %.8f\n\n", dz);

    //printf("Total %.8f\n", d);
    //d = std::chrono::duration_cast<std::chrono::microseconds>(e-s1).count() / 1000000.0;
    //printf("Kernels: %.8f\n", d);
    //printf("Kernels only exec [us]: %.8f\n", time_kernel);
    //printf("Kernels only copy [us]: %.8f\n", time_copy);

	//Store the result into a file
	FILE *fpo = fopen("result.txt","w");
	for(int i=0;i<no_of_nodes;i++)
		fprintf(fpo,"%d) cost:%d\n",i,h_cost[i]);
	fclose(fpo);
	//printf("Result stored in result.txt\n");


	// cleanup memory
	free( h_graph_nodes);
	free( h_graph_edges);
	free( h_graph_mask);
	free( h_updating_graph_mask);
	free( h_graph_visited);
	free( h_cost);
	cudaFree(d_graph_nodes);
	cudaFree(d_graph_edges);
	cudaFree(d_graph_mask);
	cudaFree(d_updating_graph_mask);
	cudaFree(d_graph_visited);
	cudaFree(d_cost);
}
