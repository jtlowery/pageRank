'''
3/5/2015
Pagerank implementation
 
TO RUN: python pageRank.py graphName alphaValue convergenceCutoffValue outFileRootName
    graphName = 'example1' or 'example2' or 'example3' are simple hardcoded examples; 'joel' is a link graph 
        generated from crawling the engr.uky.edu subdomain with Scrapy, 'stephen' was a similar link graph 
        provided by a classmate (graph file not provided)
    File 'Joel_engr.graph' must be in the same directory to run (or load path must be changed)
'''
import sys
import cPickle as pickle
import numpy
import scipy.sparse


def loadExampleGraph(graphChoice):
    '''loads and returns the appropriate graph based on command line selection
    if no valid selection errors
    '''
   
    if graphChoice == 'stephen':
        stephen = pickle.load(open('Stephen_engr.graph', 'rb'))  #load in Stephen's example graph
        return stephen
    elif graphChoice == 'joel':
         joel = pickle.load(open('Joel_engr.graph', 'rb'))  #load in my graph from scrapy crawl
         return joel
    elif graphChoice == 'example1':
        example1 = {'A': {'out': set(['B', 'C'])}, 
            'B': {'out': set(['C'])}, 
            'C': {'out': set(['A'])}, 
            'D':{'out': set(['C'])}}
        return example1
    elif graphChoice == 'example2':
        example2 = {'Home': {'out': set(['About', 'Product', 'Links'])},
            'About': {'out': set(['Home'])},
            'Product': {'out': set(['Home'])},
            'Links': {'out': set(['Home', 'ExternalSiteA', 'ExternalSiteB', 'ExternalSiteC', 'ExternalSiteD'])}}
        return example2
    elif graphChoice == 'example3':
        example3 = {'Home': {'out': set(['About', 'Product', 'Links'])},
            'About': {'out': set(['Home'])},
            'Product': {'out': set(['Home'])},
            'Links': {'out': set(['Home', 'ReviewA', 'ReviewB', 'ReviewC', 'ReviewD', 'ExternalSiteA', 
                'ExternalSiteB', 'ExternalSiteC', 'ExternalSiteD'])},
            'ReviewA': {'out': set(['Home'])},
            'ReviewB': {'out': set(['Home'])},
            'ReviewC': {'out': set(['Home'])},
            'ReviewD': {'out': set(['Home'])},
            'ExternalSiteA': {'out': set([])},
            'ExternalSiteB': {'out': set([])},
            'ExternalSiteC': {'out': set([])},
            'ExternalSiteD': {'out': set([])}}
        return example3
    else:
        print 'Please enter a valid choice (joel, stephen, example1, example2, example3)'
        sys.exit(0)


def pageRank(linkGraph, alpha, convergenceCutoff, outfileRootName):
    ''' 
    Inputs: 
        linkGraph - link graph as a matrix where column[x] corresponds to outlinks of a page x and row[x] correponds to inlinks of page x
        alpha - probability that a theoretical surfer follows a link, (1-alpha) is the probability that surfer goes to page at random
        convergenceCutoff - difference between calculations of pagerank, determines when pagerank has converged (often called error or epsilon)
        outfileRootName - provided root filename to write iterations and differences to to observe convergence
    Outputs:
        Ik - the most recently computed pagerank upon convergence
        also writes out the iteration number and difference between latest iterations of pageranks
    Based on power method described in: 
    Austin, David. "How Google Finds Your Needle in the Web's Haystack". 
    Feature Column, American Mathematical Society.
    http://www.ams.org/samplings/feature-column/fcarc-pagerank
    '''
    H, colSums = normalizeColumns(linkGraph) #normalize the matrix so nonzero columns sum to 1 and find where columns sum to 0 (dangling links, no outlinks)
    matrixSize = len(colSums)
    Ik = numpy.ones(matrixSize)/matrixSize #initialize matrix so all values start at 1/n, will converge
    Ik_plus1 = numpy.zeros(matrixSize)
    iterationCount = 0
    difference = 10000.00
    f_out = open(outfileRootName+'.iterations', 'w')
    f_out.write('iteration' + '\t' + 'difference' + '\n')
    print 'Starting PR calculation...'
    while (difference > convergenceCutoff):
        #these first two terms should always be the same for a given iteration so should only be computed one time
        alpha_oneIk = numpy.dot(((1-alpha)/matrixSize)*numpy.ones(matrixSize), Ik) #surfer random new link term
        
        #term to handle surfer dead ending in sinks
        alpha_AIk = numpy.dot((alpha * (colSums == 0)/float(matrixSize)), Ik) 
        #colSums == 0 returns a boolean array (of shape n, )
        #where every column index of a dangling link is 1
        
        for rowIndex in xrange(0, matrixSize):
            #the row at rowindex represents the inlinks for the page at that index 
            inlinksForRowIndex = H[rowIndex,:].todense() #pull the row from the csr and convert to dense form shape (1, matrixSize)
            inlinksForRowIndex = numpy.array(inlinksForRowIndex)[0,:] #convert to numpy.array in shape (matrixSize, ) to match other terms 
            alpha_HIk = numpy.dot(Ik, (inlinksForRowIndex*alpha)) #compute the term for the inlinks' contribution
            Ik_plus1[rowIndex] = alpha_HIk + alpha_AIk + alpha_oneIk #summing all the terms up
            #Ik_plus1 = alpha * H * Ik + alpha * A Ik + ((1-alpha)/matrixSize) 1 * Ik #whole formula for reference
        print Ik_plus1
        iterationCount += 1
        print '\tCurrent iteration:', iterationCount  
        Ik_plus1 = Ik_plus1 / float(sum(Ik_plus1)) #normalize so all pageranks sum to 1
        print '\tSum Ik_plus1:', sum(Ik_plus1)
        difference = sum(abs(Ik-Ik_plus1)) #compute new difference to see if converged
        Ik = Ik_plus1 #set old to new
        Ik_plus1 = numpy.zeros(matrixSize) #zero out Ik_plus1
        print '\tCurrent difference:', difference
        f_out.write( str(iterationCount) + '\t' +  str(difference) + '\n')
    f_out.close()
    return Ik
    
def buildLinkGraph(dictInlinksOutlinks):
    '''
    Input: loads in a dictionary containing link graph in form:
    'url': {'out':set(outlinks), 'in':set(inlinks)} OR 'url': {'out':set(outlinks)}
    Output: outputs the link graph  in a matrix form
    '''
    allLinks = set()
    urlToIndex = {}
    indexToUrl = {}
    pageToOutlinkIndices = {} #dict of outlink indices for each page
    posCount = 0
    print 'Building index-URL dicts and determining size...\n'
    for page, links in dictInlinksOutlinks.iteritems(): 
        #determining size of matrix and linking URLs to an index for easy construction of link graph in matrix form
        #print 'processing...', page

        #handle current page
        if page not in allLinks:
            indexToUrl[posCount] = page
            urlToIndex[page] = posCount
            posCount += 1
            allLinks.add(page) #add current page to set
        
        #handle current page's outlinks
        outlinks = links['out']
        #print type(outlinks)
        assert type(outlinks) == set
        outlinkIndices = set()
        for link in outlinks:
            if link not in allLinks:
                indexToUrl[posCount] = link
                urlToIndex[link] = posCount
                posCount += 1
            outlinkIndices.add(urlToIndex[link]) #build set of indices for outlinks for current page
        allLinks.update(outlinks)
        pageToOutlinkIndices[page] = outlinkIndices
        #print page, pageToOutlinkIndices[page]

    n = len(allLinks) #number of links -- matrix size n,n
    print 'size of allLinks: ', n, '\n'
        
    matrixLinkGraph  = numpy.zeros((n,n), dtype=float) # create linkgraph as an initial n,n matrix of zeros since sparse
    print 'Setting values for outlinks in link graph matrix...\n'
    for page, outlinks in pageToOutlinkIndices.iteritems():
        #setting values in matrix
        currentPageIndex = urlToIndex[page] #find the index for the current page, this will be our column index
        #so each column will represent the outlinks of a page
        for outlinkIndex in outlinks:
            matrixLinkGraph[outlinkIndex, currentPageIndex] = 1
    #print matrixLinkGraph[:,2]
    return matrixLinkGraph, indexToUrl
    
def normalizeColumns(matrixIn):
    matrixIn = scipy.sparse.csr_matrix(matrixIn, dtype=numpy.float) #build a compressed sparse row matrix
    colSums = numpy.array(matrixIn.sum(axis = 0)) #sums by columns, returns shape 1, n array
    colSums = colSums[0, :] #converts to array of shape n,
    nonZeroRowIndices, nonZeroColIndices = matrixIn.nonzero() #returns 2 arrays of the indices of nonzero elements
    matrixIn.data = matrixIn.data/colSums[nonZeroColIndices] #divide nonzero columns by the column sums
    return matrixIn, colSums
    
if __name__ == '__main__':
    if (len(sys.argv) != 5):
        print 'Errror! Usage: python pageRank.py graphName alphaValue convergenceCutoffValue outFileRootName' 
    examplegraph = sys.argv[1]
    alpha = float(sys.argv[2])
    convergenceCutoff = float(sys.argv[3])
    outfileRootName = sys.argv[4]
    dictLinks = loadExampleGraph(examplegraph) 
    linkGraph, indexToUrl = buildLinkGraph(dictLinks)
    
    print linkGraph
    PR = pageRank(linkGraph, alpha, convergenceCutoff, outfileRootName)

    summed = 0
    rankedList = []
    for num, pagerank in enumerate(PR):
        rankedList.append((indexToUrl[num], pagerank))
        #print type(rankedList)
        summed += pagerank
    print 'Summed PR: ', summed
    f_out = open(outfileRootName+'.pageranks', 'w') 
    f_out.write('Page' + '\t' + 'Pagerank' + '\n')
    for x in sorted(rankedList, key=lambda tuple:tuple[1], reverse = True): #writing out pagerank and corresponding url in descending PR
        f_out.write(x[0] + '\t' + str(x[1]) + '\n')
    f_out.close()