/*
 *	This Header file stores the mapping of border vertices to the 
 *	vertices in the graph. Use this to reduce size of the border vectors.
 *	
 *	The mappings are not specific to actual vertex IDs in the entire graph or the
 *	vertex IDs in a partition. It can be used for both.
 *
 *	But its advised to use as partition vertex ID
 *	
 *	For actual vertex Id to partition vertex id refere PartitionVertexMapping.h
 *
 *	Keep the border vertex values (distance, sigma, etc) seperate from
 *	this mapping.
 *	
 *	By:-
 *	Ashirbad Mishra
 *
 * */

// Struct to store the mapping of border node
// to the vertex ID
struct BorderToVertex
{
	unsigned vertexID;
};

// Struct to store the mapping of vertex node
// // to the border ID
struct VertexToBorder
{
	unsigned borderID;
};

// Struct store the complete mapping
// of a border ID to vertex ID
// and vice versa
struct BorderVertxMapping
{
	/* Members */
	// Used to assign borderIDs
	static unsigned seqNum=0;
	unsigned numVertices;	
	unsigned numBorders;

	// Each complete struct will contain
	// a forward and backward mapping
	struct VertexToBorder * vertexToBorder;
	struct BorderToVertex * borderToVertex;

	/* Methods */
	BorderVertxMapping (unsigned numVertices, unsigned numBorders);
	~BorderVertxMapping();

	//only use this
	unsigned getVertexID( unsigned borderID );
	unsigned getBorderID( unsigned vertexID );

	//never use this
	void 	 setVertexIDToBorder( unsigned vertexID, unsigned borderID );
	void 	 setBorderIDToVertex( unsigned vertexID, unsigned borderID );
	
};

BorderVertexMapping::BorderVertxMapping( unsigned numVertices, unsigned numBorders, unsigned *border )
{
	// Set no of vertices and borders
	this.numVertices = numVertices;
	this.numBorders = numBorders

	// Initializes the struct by allocating memory for
	// border to vertex mapping 
	// vertex to border mapping
	this.vertexToBorder = ( struct VertexToBorder * ) malloc ( numVertices * sizeof( struct VertexToBorder ) );
	this.borderToVertex = ( struct BorderToVertex * ) malloc ( numBorders * sizeof( struct BorderToVertex ) );

	// Maps the vertices to borders
	// borderIDs are allocated incrementally starting with 0 
	// based on seqNum and the border array passed to it
	// It is assumed that the border[] array size is equal to numVertices,
	// i.e each entry of border array contains info for each vertex
	for( int i = 0; i < numVertices; i++ )
	{
		// for non-zero value of border[], its a border node hence store the mapping
		// else store the mapping as -1 that vertex
		if( border[ i ] != 0 )
		{
			vertexToBorder[i].borderID = seqNum;
			borderToVertex[seqNUM].vertexID = i;
			++seqNum;
		}
		else
		{
			vertexToBorder[i].borderID = -1;
		}
	}
}

BorderVertexMapping::~BorderVertxMapping()
{
	free( this.vertexToBorder )
	free( this.borderToVertex )
}

unsigned BorderVertexMapping::getVertexID( unsigned borderID )
{
	return( this.borderToVertex[borderID].vertexID );
}

unsigned BorderVertexMapping::getBorderID( unsigned vertexID )
{
	return( this.vertexToBorder[vertexID].borderID );
}

void BorderVertexMapping::setVertexIDToBorder( unsigned vertexID, unsigned borderID )
{
	this.borderToVertex[borderID].vertexID = vertexID;
}

void BorderVertexMapping::setBorderIDToVertex( unsigned vertexID, unsigned vertexID )
{
	this.vertexToBorder[vertexID].borderID = borderID;
}

