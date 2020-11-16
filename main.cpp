#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <float.h>
#include <math.h>
#include <iostream>
#include "libarff/arff_parser.h"
#include "libarff/arff_data.h"
#include <algorithm>
#include <tuple>

using namespace std;

#define BUFFERSIZE sizeof(int) // TODO maybe size of an int
#define TAG 0



int* KNN(ArffData* dataset, int k_value)
{
    // predictions is the array where you have to return the class predicted (integer) for the dataset instances
    int* predictions = (int*)malloc(dataset->num_instances() * sizeof(int));
    
    // The following two lines show the syntax to retrieve the attribute values and the class value for a given instance in the dataset
    // float attributeValue = dataset->get_instance(instanceIndex)->get(attributeIndex)->operator float();
    // int classValue =  dataset->get_instance(instanceIndex)->get(dataset->num_attributes() - 1)->operator int32();
    
    // Implement the KNN here, fill the predictions array

    // allocate space of array of tuples of distances and indices to sort based on distances and to be able to parse for neighbors
    tuple<int, float>* ind_dist = (tuple<int, float>*)malloc(dataset->num_instances() * sizeof(tuple<int, float>));

    // for each instance in the dataset
    for(int i=0; i < dataset->num_instances(); i++)
    {
        // target all other instances in the dataset
        for(int j=0; j < dataset->num_instances(); j++)
        {
            // if instance(i) == instance(j) the distance will be 0 for accuracy
            if(i == j)
            {
                ind_dist[j] = tuple<int, float>(j, FLT_MAX);
                continue;
            }

            // calculating euclidean distance between two vectors in n space for all instances
            float distance = 0;

            for(int k=0; k < dataset->num_attributes() - 1; k++)
            {
                float difference = dataset->get_instance(i)->get(k)->operator float() - dataset->get_instance(j)->get(k)->operator float();
                distance += difference * difference;
            }
            distance = sqrt(distance);

            // calculate mode to find k
            ind_dist[j] = tuple<int, float>(j, distance);

            // cout << get<0>(ind_dist[j]) << " : ";
            // cout << get<1>(ind_dist[j]) << "\n"; 
        }
        
        // sort based on distance, now indices are out of order but the first elements in the array are the closest
        sort(ind_dist, ind_dist + dataset->num_instances(), 
            [](tuple<int,float> x, tuple<int,float> y)
            {
                return get<1>(x) < get<1>(y);
            });

        // get k neighbors
        int neighbors [k_value];
        // int distances [k_value];

        for(int m = 0; m < k_value; m++)
        {
            neighbors[m] = get<0>(ind_dist[m]); // get the closest indices and put the in neighbors
            // printf("neighbors at %d are %d\n", i, neighbors[i]);
            // distances[m] = get<1>(ind_dist[m]); // this ends up being 0.000

            // printf("neighbors indices: %d distances: %f\n", neighbors[m], distances[m]);
        }

        // map neighbors distances with classes based on the indices -- similar to dictionaries in python
        int classes [k_value]; // per alwin this should  be an array to save memory then just use indexes

        // gets all of the classes for the k nearest neighbors
        for (int m = 0; m < k_value; m++)
        {
            classes[m] = dataset->get_instance(neighbors[m])->get(dataset->num_attributes()-1)->operator int32();
            // printf("classes: %d ",classes[m]);
        }

        // dictionary to hold count and class
        map<int, int> dictionary;

        int count = -1;
        int mode = -1;
        int class_value;

        // find the mode within the dictionary
        for (int m = 0; m < k_value; m++)
        {
            class_value = classes[m];
            dictionary[class_value]++; // increment count
            if (dictionary[class_value] > count) // set mode to count with highest val
            {
                count = dictionary[class_value];
                mode = class_value;
                // printf("mode %d\n", mode);
            }
        }



        // get mode or max of an array based on count
        // int count[12];
        
        // // set all numbers in count to 0
        // for(int m = 0; m < k;m++)
        // {
        //     count[m]=0;
        // }

        // // increase count for every occurence where dictionary[m] is found
        // for(int m = 0;m < k; m++)
        // {
        //     count[classes[m]]++; 
        //     // printf("count is %d\n", count[dictionary[m]]);
        // }

        // // cout << count[i] << " ";

        // int mode = 0;

        // for (int q = 0; q < k; q++)
        // {
        //     // printf("count[q] == %d\n", count[q]);
        //     if (count[q] > mode)
        //     {
        //         mode = q;
        //         // printf("mode %d\n", mode);
        //     }
        // }
    
        // printf("mode is %d\n", mode);

        predictions[i] = mode;

        // cout << dataset->num_classes() << " : dataset num_classes\n";
        // cout << "size of count:" << sizeof(count);

        // printf("dictionary %d\n", dictionary[i]);
        // predictions[i]=dictionary[i];
        // printf("pred %d\n", predic"tions[i]);
    }
    // printf("who th knows%d \n", *predictions);
    // geTODOTODOt class/last element in dataset babased on 

    free(ind_dist);

    return predictions;
}

























// parallelized knn
int* KNN_MPI(ArffData* dataset, int k_value, int rank, int size)
{
    int* pred = (int*)malloc(dataset->num_instances() * sizeof(int));

    int z;

    MPI_Status status[dataset->num_instances()];
    MPI_Request send_request[dataset->num_instances()];
    MPI_Request receive_request[dataset->num_instances()];

    // int count = (dataset->num_instances()) / size;
    // int remainder = (dataset->num_instances()) % size;
    int count = ceil( (float)dataset->num_instances() / (float)(size));
    int start, stop;

    // cout << "remainder: " << remainder << "\n";
    // if (rank < remainder) 
    // {
    //     start = rank * (count + 1);
    //     stop = start + count;

    //     cout << "start: " << start << " stop: " << stop << "\n";

    //     for(z=start; z <= stop; z++)
    //     {
    //         // cout << "send loop z: "<< z << " " << blah << " \n";
    //         MPI_Isend(&blah, 1, MPI_INT, 0, TAG, MPI_COMM_WORLD, &send_request[z]); // in the send buffer 
    //     }
    
    // }else{
    //     start = rank * count + remainder;
    //     stop = start + (count - 1);

        start = rank * count;
        stop = ((rank + 1) * count) - 1 > dataset->num_instances() - 1 ? dataset->num_instances() - 1: ((rank + 1) * count) - 1;

        // cout << "start: " << start << " stop: " << stop << "\n";
        tuple<int, float>* ind_dist = (tuple<int, float>*)malloc(dataset->num_instances() * sizeof(tuple<int, float>));

        for(z=start; z <= stop; z++)
        {

            for(int j=0; j < dataset->num_instances(); j++)
            {
                if(z == j)
                {
                    ind_dist[j] = tuple<int, float>(j, FLT_MAX);
                    continue;
                }

                // calculating euclidean distance between two vectors in n space for all instances
                float distance = 0;

                for(int k=0; k < dataset->num_attributes() - 1; k++)
                {
                    float difference = dataset->get_instance(z)->get(k)->operator float() - dataset->get_instance(j)->get(k)->operator float();
                    distance += difference * difference;
                }
                distance = sqrt(distance);

                // calculate mode to find k
                ind_dist[j] = tuple<int, float>(j, distance);
            }

            sort(ind_dist, ind_dist + dataset->num_instances(), 
                [](tuple<int,float> x, tuple<int,float> y)
                {
                    return get<1>(x) < get<1>(y);
                });

            // get k neighbors
            int neighbors [k_value];

            for(int m = 0; m < k_value; m++)
            {
                neighbors[m] = get<0>(ind_dist[m]);
            }

            int classes [k_value];

            // gets all of the classes for the k nearest neighbors
            for (int m = 0; m < k_value; m++)
            {
                classes[m] = dataset->get_instance(neighbors[m])->get(dataset->num_attributes()-1)->operator int32();
            }

            // dictionary to hold count and class
            map<int, int> dictionary;

            int count = -1;
            int mode = -1;
            int class_value;

            // find the mode within the dictionary
            for (int m = 0; m < k_value; m++)
            {
                class_value = classes[m];
                dictionary[class_value]++;
                if (dictionary[class_value] > count)
                {
                    count = dictionary[class_value];
                    mode = class_value;
                }
            }

            // cout << "mode: " << mode << endl;
            // cout << "send loop z: "<< z << " " << blah << " \n";
            MPI_Isend(&mode, 1, MPI_INT, 0, z, MPI_COMM_WORLD, &send_request[z]); // in the send buffer 
        }

    // MPI_Waitall(start - stop, send_request, status); // MAYBE IGNORE BROKE THIS
    // MPI_Barrier(MPI_COMM_WORLD); 
    if(rank == 0)
    {
        for(int f = 0; f < dataset->num_instances(); f++)
        {
            // cout << "recv loop f: "<< f << " " << blah << endl;
            MPI_Irecv(&pred[f], 1, MPI_INT, MPI_ANY_SOURCE, f, MPI_COMM_WORLD, &receive_request[f]);
        }
        MPI_Waitall(dataset->num_instances(), receive_request, status);
    }
    
    MPI_Barrier(MPI_COMM_WORLD);

    return pred;
}































int* computeConfusionMatrix(int* predictions, ArffData* dataset)
{
    int* confusionMatrix = (int*)calloc(dataset->num_classes() * dataset->num_classes(), sizeof(int)); // matrix size numberClasses x numberClasses
    
    for(int i = 0; i < dataset->num_instances(); i++) // for each instance compare the true class and predicted class
    {
        int trueClass = dataset->get_instance(i)->get(dataset->num_attributes() - 1)->operator int32();
        int predictedClass = predictions[i];
        
        confusionMatrix[trueClass*dataset->num_classes() + predictedClass]++;
    }
    
    return confusionMatrix;
}

float computeAccuracy(int* confusionMatrix, ArffData* dataset)
{
    int successfulPredictions = 0;
    
    for(int i = 0; i < dataset->num_classes(); i++)
    {
        successfulPredictions += confusionMatrix[i*dataset->num_classes() + i]; // elements in the diagonal are correct predictions
    }
    
    return successfulPredictions / (float) dataset->num_instances();
}

int main(int argc, char *argv[])
{
    int k_value, rank, size;

    if(argc != 3)
    {
        cout << "Usage: ./main datasets/datasetFile.arff k_value" << endl;
        exit(0);
    }

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); // process id
    MPI_Comm_size(MPI_COMM_WORLD, &size); // numtasks
    
    // set k value
    k_value = atoi(argv[2]);

    // Open the dataset
    ArffParser parser(argv[1]);
    ArffData *dataset = parser.parse();
    struct timespec start, end;

    if(rank == 0)
    {  
        clock_gettime(CLOCK_MONOTONIC_RAW, &start);

        // Get the class predictions
        int* predictions = KNN(dataset, k_value);
    
        // Compute the confusion matrix
        int* confusionMatrix = computeConfusionMatrix(predictions, dataset);
        // Calculate the accuracy
        float accuracy = computeAccuracy(confusionMatrix, dataset);
    
        clock_gettime(CLOCK_MONOTONIC_RAW, &end);
        uint64_t diff = (1000000000L * (end.tv_sec - start.tv_sec) + end.tv_nsec - start.tv_nsec) / 1e6;
  
        printf("The KNN classifier for %lu instances required %llu ms CPU time, accuracy was %.4f\n", dataset->num_instances(), (long long unsigned int) diff, accuracy);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    ////////////// MPI version below
    
    clock_gettime(CLOCK_MONOTONIC_RAW, &start);

    int *pred_MPI = KNN_MPI(dataset, k_value, rank, size);
    
    if(rank == 0)
    {
        clock_gettime(CLOCK_MONOTONIC_RAW, &end);
        uint64_t diff_MPI = (1000000000L * (end.tv_sec - start.tv_sec) + end.tv_nsec - start.tv_nsec) / 1e6;
        
        int* confusionMatrix_MPI = computeConfusionMatrix(pred_MPI, dataset);
        float accuracy_MPI = computeAccuracy(confusionMatrix_MPI, dataset);

        printf("The KNN classifier MPI for %lu instances required %llu ms CPU time, accuracy was %.4f\n", dataset->num_instances(), (long long unsigned int) diff_MPI, accuracy_MPI);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
}