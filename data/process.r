# install if necessary
# if (!requireNamespace("BiocManager", quietly = TRUE))
#     install.packages("BiocManager", repos='http://cran.us.r-project.org')

# BiocManager::install("GenomicRanges")


library(GenomicRanges)


# install if necessary
# if (!requireNamespace("BiocManager", quietly = TRUE))
#     install.packages("BiocManager", repos='http://cran.us.r-project.org')

# BiocManager::install("InteractionSet")


library(InteractionSet)


datapath = '/N/slate/bizi/DeepChromeHiC/data'
# datapath = 'E:/iu/DeepChromeHiC/data'
# Set working path
setwd(datapath)


all_gen = list.files( getwd(), pattern = '*.rda' )


for (gen in all_gen)
{
	anchor = load(gen)


	normAnchor <- function(anchor, ext = 5000)
	{
	  mid = round( (start(anchor) + end(anchor))/2 )
	  start(anchor) = mid - ext
	  end(anchor) = mid + ext
	  anchor
	}


	seqAnchor <- function(anchor)
	{
	  seqs = getSeq(BSgenome.Hsapiens.UCSC.hg19, anchor)
	  if (length( grep('N', seqs) ) > 0)
	  {
		seqs = DNAStringSet( gsub('N', sample(c('A', 'C', 'G', 'T'), 1), seqs) )
	  }
	  seqs
	}


	# install if necessary
	# if (!requireNamespace('BiocManager', quietly = TRUE))
	#     install.packages('BiocManager', repos='http://cran.us.r-project.org')

	# BiocManager::install('BSgenome.Hsapiens.UCSC.hg19')


	library(BSgenome.Hsapiens.UCSC.hg19)


	anchor1.pos.norm = normAnchor(anchor1.pos)
	seq.anchor1.pos = seqAnchor(anchor1.pos.norm)
	
	
	# print(length(seq.anchor1.pos))
	if (length(seq.anchor1.pos) > 2000)
	{
		print( c(gen, length(seq.anchor1.pos)) )
	}
	
	
	if ( length(seq.anchor1.pos) > 2000 )
	{
		anchor2.pos.norm = normAnchor(anchor2.pos)
		seq.anchor2.pos = seqAnchor(anchor2.pos.norm)

		anchor1.neg.norm = normAnchor(anchor1.neg)
		seq.anchor1.neg2 = seqAnchor(anchor1.neg.norm)

		anchor2.neg.norm = normAnchor(anchor2.neg)
		seq.anchor2.neg2 = seqAnchor(anchor2.neg.norm)
		
		
		# Delete the .rda in the string
		name = paste0(strsplit(gen, '.', fixed = TRUE)[[1]][1], '.',
		strsplit(gen, '.', fixed = TRUE)[[1]][2], '/')
		# print(name)
		
		
		# new folder
		dir.create(name)


		write.table(as.data.frame(seq.anchor1.pos), file = paste0(name, 'seq.anchor1.pos.txt'), row.names = F, col.names = F, quote = F)
		write.table(as.data.frame(seq.anchor1.neg2), file = paste0(name, 'seq.anchor1.neg2.txt'), row.names = F, col.names = F, quote = F)

		write.table(as.data.frame(seq.anchor2.pos), file = paste0(name, 'seq.anchor2.pos.txt'), row.names = F, col.names = F, quote = F)
		write.table(as.data.frame(seq.anchor2.neg2), file = paste0(name, 'seq.anchor2.neg2.txt'), row.names = F, col.names = F, quote = F)
	}
	else
	{
		# print('too short, skip ...')
	}
}