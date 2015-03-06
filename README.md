##Simple pageRank Implementation

Date: 3/5/2015

To Run: python pageRank.py graphName alphaValue convergenceCutoffValue outFileRootName
    
	graphName = 'example1' or 'example2' or 'example3' are simple hardcoded examples; 'joel' is a link graph 
        generated from crawling the engr.uky.edu subdomain with Scrapy, 'stephen' was a similar link graph 
        provided by a classmate (graph file not provided here)
    
	alpha Value: should be between 1 and 0, 0.85 suggested
	
	convergenceCutoffValue: should be much less than 1. 0.0001 suggested.
	
	File 'Joel_engr.graph' must be in the same directory to run (or load path must be changed)