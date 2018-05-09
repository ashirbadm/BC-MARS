/*
 *	This file contains the mapping from the actual ID of a vertex
 *	in the graph to the ID of a vertex in a partition.
 *
 *	Use this to reduce the vertex vector sizes
 *	
 *	By:-
 *	Ashirbad Mishra
 *
 * */

struct GraphToPartitionVID
{
	unsigned partitionVID;
};

struct PartitionToGraphVID
{
	unsigned graphVID;
};

struct PartitionVertexMapping 
{
	struct GraphToPartitionVID *graphToPartitionVID;
	struct PartitionToGraphVID **partitionToGraphVID;
	
	//  Param: 
	//  Number of partitions of graph
	//  Number of nodes per partition of graph
	//  Number of borders per partition of graph
	//  Partition array that denotes to which partition each vertex belongs to
	//  Number of nodes of entire graph
	PartitionVertexMapping( int numParts, unsigned *nodeCount, unsigned *borderCount, unsigned *part, unsigned numGraphNodes );
	~PartitionVertexMapping();
	unsigned getGraphVertexID( unsigned partitionVertexID, int partitionID );
	unsigned getPartitionVertexID( unsigned graphVertexID );
};

PartitionVertexMapping::PartitionVertexMapping( int numParts, unsigned *nodeCount, unsigned *borderCount, unsigned *part, unsigned numGraphNodes )
{
	// Initialize ID arrays
	this.graphToPartitionVID = (struct GraphToPartitionVID *) malloc(sizeof(struct GraphToPartitionVID)*numGraphNodes);
	this.partitionToGraphVID = (struct PartitionToGraphVID **) malloc(sizeof(struct PartitionToGraphVID *) * numParts);

	// Initialize seq number for assigning partition VIDs
	int partitionSeqNum [numParts];
	//Calculate the numNodes in each partition
	int numNodes[numParts];
	for(int partitionIndex=0 ; partitionIndex < numParts ; partitionIndex++)
	{	
		partitionSeqNum[partitionIndex] = 0;
		// Assign its own #nodes
		numNodes[partitionIndex] = nodeCount[partitionIndex];
		int subIndex = 0;
		while( subIndex < numParts )
		{	
			if(subIndex != partitionIndex)
			{
				// Assign other partition's #border nodes to current partition
				numNodes[partitionIndex] += borderCount[1-subIndex];
			}
			subIndex++;
		}
		this.partitionToGraphVID[partitionIndex] = (struct PartitionToGraphVID *) malloc(sizeof(struct PartitionToGraphVID) * numNodes[partitionIndex]);
	}

	// Iterate over each graph vertex id and 
	// assign it a partition vertex id based on which
	// partition it belongs to
	for( unsigned graphVID =0 ; graphVID < numGraphNodes; graphVID++)
	{
		// Assign the partition VID for a particular vertex in graph
		int ownPartitionID = part[graphVID];
		this.graphToPartitionVID[graphVID].partitionVID = partitionSeqNum[ownPartitionID];
		this.partitionToGraphVID[ownPartitionID][partitionSeqNum[ownPartitionID]].graphVID = graphVID;
		++partitionSeqNum[ownPartitionID];

		// If vertex is a border then assign a partition VID
		// for every other partition
		if( border[graphVID] )
		{
			for(int otherPartitionID=0 ; otherPartitionID < numParts ; otherPartitionID++)
			{
				if( partitionIndex != partitionID )
				{
					this.graphToPartitionVID[graphVID].partitionVID = partitionSeqNum[otherPartitionID];
					this.partitionToGraphVID[otherPartitionID][partitionSeqNum[otherPartitionID]].graphVID = graphVID;
					++paritionSeqNum[otherPartitionID];
				}
			}
		}
	}
}

unsigned getGraphVertexID( unsigned partitionVertexID, int partitionID )
{
	return( this.partitionToGraphVID[partitionID][partitionVertexID].graphVID );
}
unsigned getPartitionVertexID( unsigned graphVertexID )
{
	return( this.graphToPartitionVID[graphVertexID].partitionVID );
}

