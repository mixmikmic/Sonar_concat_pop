# Once we've trained a model, we might want to better understand what sequence motifs the first convolutional layer has discovered and how it's using them. Basset offers two methods to help users explore these filters.
# 
# You'll want to double check that you have the Tomtom motif comparison tool for the MEME suite installed. Tomtom provides rigorous methods to compare the filters to a database of motifs. You can download it from here: http://meme-suite.org/doc/download.html
# 
# To run this tutorial, you'll need to either download the pre-trained model from https://www.dropbox.com/s/rguytuztemctkf8/pretrained_model.th.gz and preprocess the consortium data, or just substitute your own files here:
# 

model_file = '../data/models/pretrained_model.th'
seqs_file = '../data/encode_roadmap.h5'


# First, we'll run basset_motifs.py, which will extract a bunch of basic information about the first layer filters. The script takes an HDF file (such as any preprocessed with preprocess_features.py) and samples sequences from the test set. It sends those sequences through the neural network and examines its hidden unit values to describe what they're doing.
# 
# * -m specifies the Tomtom motif database: CIS-BP Homo sapiens database by default.
# * -s specifies the number of sequences to sample. 1000 is fast and sufficient.
# * -t asks the script to trim uninformative positions off the filter ends.
# 

import subprocess

cmd = 'basset_motifs.py -s 1000 -t -o motifs_out %s %s' % (model_file, seqs_file)
subprocess.call(cmd, shell=True)


# Now there's plenty of information output in motifs_out. My favorite way to get started is to open the HTML file output by Tomtom's comparison of the motifs to a database. It displays all of the motifs and their database matches in a neat table.
# 
# Before we take a look though, let me describe where these position weight matrices came from. Inside the neural network, the filters are reprsented by real-valued matrices. Here's one:
# 

# actual file is motifs_out/filter9_heat.pdf

from IPython.display import Image
Image(filename='motifs_eg/filter9_heat.png') 


# Although it's matrix of values, this doesn't quite match up with the conventional notion of a position weight matrix that we typically use to represent sequences motifs in genome biology. To make that, basset_motifs.py pulls out the underlying sequences that activate the filters in the test sequences and passes that to weblogo.
# 

# actual file is motifs_out/filter9_logo.eps

Image(filename='motifs_eg/filter9_logo.png') 


# The Tomtom output will annotate the filters, but it won't tell you how the model is using them. So alongside it, let's print out a table of the filters with the greatest output standard deviation over this test set. Variance correlates strongly with how influential the filter is for making predictions.
# 
# The columns here are:
# 1. Index
# 2. Optimal sequence
# 3. Tomtom annotation
# 4. Information content
# 5. Activation mean
# 6. Activation standard deviaion
# 

get_ipython().system('sort -k6 -gr motifs_out/table.txt | head -n20')


# As I discuss in the paper, unannotated low complexity filters tend to rise to the top here because low order sequence complexity influence accessibility.
# 
# The Tomtom output HTML is here:
# 

get_ipython().system('open motifs_out/tomtom/tomtom.html')


# The other primary tool that we have to understand the filters is to remove the filter from the model and assess the impact. Rather than truly remove it and re-train, we can just nullify it within the model by setting all output from the filter to its mean. This way the model around it isn't drastically affected, but there's no information flowing through.
# 
# This analysis requires considerably more compute time, so I separated it into a different script. To give it a sufficient number of sequences to obtain a good estimate influence, I typically run it overnight. If your computer is using too much memory, decrease the batch size. I'm going to run here with 1,000 sequences, but I used 20,000 for the paper.
# 
# To get really useful output, the script needs a few additional pieces of information:
# * -m specifies the table created by basset_motifs.py above.
# * -s samples 2000 sequences
# * -b sets the batch_size to 500
# * -o is the output directory
# * --subset limits the cells displayed to those listed in the file.
# * -t specifies a table where the second column is the target labels.
# * --weight, --height, --font adjust the heatmap
# 

cmd = 'basset_motifs_infl.py'
cmd += ' -m motifs_out/table.txt'
cmd += ' -s 2000 -b 500'
cmd += ' -o infl_out'
cmd += ' --subset motifs_eg/primary_cells.txt'
cmd += ' -t motifs_eg/cell_activity.txt'
cmd += ' --width 7 --height 40 --font 0.5'
cmd += ' %s %s' % (model_file, seqs_file)

subprocess.call(cmd, shell=True)


# Once you've trained a model, you might like to get more information about how it performs on the various targets you asked it to predict.
# 
# To run this tutorial, you'll need to either download the pre-trained model from https://www.dropbox.com/s/rguytuztemctkf8/pretrained_model.th.gz and preprocess the consortium data, or just substitute your own files here:
# 

model_file = '../data/models/pretrained_model.th'
seqs_file = '../data/encode_roadmap.h5'


# As long as your HDF5 file has test data set aside, run:
# 

import subprocess

cmd = 'basset_test.lua %s %s test_out' % (model_file, seqs_file)
subprocess.call(cmd, shell=True)


# In the output directory, you'll find a table specifying the AUC for each target.
# 

get_ipython().system('head test_eg/aucs.txt')


# We can also make [receiver operating characteristic curves](https://en.wikipedia.org/wiki/Receiver_operating_characteristic) for each target with the following command.
# 

targets_file = '../data/sample_beds.txt'

cmd = 'plot_roc.py -t %s test_out' % (targets_file)
subprocess.call(cmd, shell=True)


# actual file is test_out/roc1.pdf

from IPython.display import Image
Image(filename='test_eg/roc1.png')


# In this tutorial, we'll walk through downloading and preprocessing the compendium of ENCODE and Epigenomics Roadmap data.
# 

# This part won't be very iPython tutorial-ly...
# 
# First cd in the terminal over to the data directory and run the script *get_dnase.sh*.
# 
# That will download all of the BED files from ENCODE and Epigenomics Roadmap. Read the script to see where I'm getting those files from. Perhaps there will be more in the future, and you'll want to manipulate the links.
# 

# Once that has finished, we need to merge all of the BED files into one BED and an activity table.
# 
# I typically use the -y option to avoid the Y chromosome, since I don't know which samples sequenced male or female cells.
# 
# I'll use my default of extending the sequences to 600 bp, and merging sites that overlap by more than 200 bp. But you might want to edit these.
# 

get_ipython().system('cd ../data; preprocess_features.py -y -m 200 -s 600 -o er -c genomes/human.hg19.genome sample_beds.txt')


# To convert the sequences to the format needed by Torch, we'll first convert to FASTA.
# 

get_ipython().system('bedtools getfasta -fi ../data/genomes/hg19.fa -bed ../data/er.bed -s -fo ../data/er.fa')


# Finally, we convert to HDF5 for Torch and set aside some data for validation and testing.
# 
# -r permutes the sequences.
# -c informs the script we're providing raw counts.
# -v specifies the size of the validation set.
# -t specifies the size of the test set.
# 

get_ipython().system('seq_hdf5.py -c -r -t 71886 -v 70000 ../data/er.fa ../data/er_act.txt ../data/er.h5')


# And you're good to go!
# 

# Perhaps Basset's most useful capability right now is to annotate noncoding genomic variants with their influence on functional properties like accessibility.
# 
# In a typical scenario, a researcher will be interested in a region that a GWAS has associated with a phenotype. In this region, there will be lead SNP that was measured in the study, and a set of nearby SNPs in linkage disequilibrium with it. A priori, the researcher cannot know which SNP(s) are truly associated with the phenotype.
# 
# The first and most important advice I can offer here is to make sure that Basset has been trained to predict relevant functional properties. Genome accessibility is a great choice because it captures a variety of regulatory events. However, you'll learn more about the SNP by training Basset on the most specific and relevant cell types that you can. See my preprocessing tutorials for advice on adding data to the ENCODE and Epigenomics Roadmap compendiums.
# 
# Here though, we'll just use the compendium. To run this tutorial, you'll need to either download the pre-trained model from https://www.dropbox.com/s/rguytuztemctkf8/pretrained_model.th.gz or substitute your own file here:
# 

model_file = '../data/models/pretrained_model.th'


# Using that trained model, *basset_sad.py* will compute SNP Accessibility Difference (SAD) profiles for a set of SNPs given in VCF format.
# 
# We'll demonstrate on an interesting SNP that I encountered in my explorations. The lead SNP [rs13336428 is associated with bone mineral density](http://www.ncbi.nlm.nih.gov/pubmed/22504420?dopt=Abstract). However, it's in linkage disequilibrium with rs67284550. I put both SNPs in *sad_eg/rs13336428.vcf*.
# 

get_ipython().system('cat sad_eg/rs13336428.vcf')


# Note that I've modified the VCF format here. The true format will still work fine. But, you can also borrow columns 6 and 7 to store some additional information about the SNP to aid downstream interpretation.
# 
# In column 6, I'm storing the lead index SNP, allowing you to group SNPs into their ambiguous haploblocks. Here, they both map to the lead SNP rs13336428 and reference its association to bone mineral density. But you can place lots of linked sets of SNPs into one file. Basset will process them all, but maintain their linkage in the table and plots.
# 
# In column 7, I'm storing a score associated with each SNP, that will be plotted next to it. Here, I've scored each SNP with its PICS probability of causality. [PICS uses fine mapping to model the GWAS signal with a better model](http://www.nature.com.ezp-prod1.hul.harvard.edu/nature/journal/v518/n7539/abs/nature13835.html). Thus, it's orthogonal to any functional analysis and useful as a barometer.
# 
# Then we run the script as follows:
# * -l specifies the input sequence length, which for this model is 600 bp
# * -i specifies that column 6 of the VCF has been co-opted to store the index SNP.
# * -s specifies that column 7 of the VCF has been co-opted to store a score.
# * -t specifies the target labels, which allows for more informative plots.
# 

import subprocess

targets_file = 'sad_eg/sample_beds.txt'

cmd = 'basset_sad.py -l 600 -i -o sad -s -t %s %s sad_eg/rs13336428.vcf' % (targets_file, model_file)
subprocess.call(cmd, shell=True)


# In the output directory, there will be a plot for every index SNP that shows the predicted accessibility across cell types, compared to the scores.
# 
# Here, we clearly see that rs13336428 has a far more substantial affect on accessibility than rs67284550.
# 

# actual file is sad/sad_Bone_mineral-rs13336428-A_heat.pdf

from IPython.display import Image
Image(filename='sad_eg/sad_Bone_mineral-rs13336428-A_heat.png')


# There will also be a table with the following columns:
# 
# 1. SNP RSID
# 2. Index SNP
# 3. Score
# 4. Reference allele
# 5. Alternative allele
# 6. Target
# 7. Reference prediction
# 8. Alternative prediction
# 9. SAD
# 
# This can help you zero in on the most affected targets.
# 

get_ipython().system(' sort -k9 -g -r  sad/sad_table.txt | head')


# To better understand what the model is seeing in the region around these SNPs, we might also perform an in silico saturated mutagenesis.
# 
# Since we'd really like to see both alleles together, I wrote a modified version called *basset_sat_vcf.py* to which we can provide a VCF file as input.
# 

cmd = 'basset_sat_vcf.py -t 6 -o sad/sat %s sad_eg/rs13336428.vcf' % model_file 
subprocess.call(cmd, shell=True)


# With the G allele, we see potential, but no activity.
# 

# actual file is sad/sat/rs13336428_G_c6_heat.pdf

Image(filename='sad_eg/rs13336428_G_c6_heat.png')


# But mutated to an A, a binding site for the AP-1 complex emerges. Just to its right, a potential PAX family motif gains relevance. Further to the right, another AP-1 motif adds to the predicted accessibility.
# 

# actual file is sad/sat/rs13336428_A_c6_heat.pdf

Image(filename='sad_eg/rs13336428_A_c6_heat.png')


# The mutation transform the regulatory potential of this site, and this analysis brings us one big step closer to figuring out how!
# 

get_ipython().magic('pylab inline')


import random, time
import numpy as np
from sklearn import preprocessing
from sklearn.decomposition import PCA, FastICA
from sklearn.manifold import TSNE, MDS, Isomap


# Let's load in the data to a panda DataFrame. A sample of the data is often better here so the routines run faster.
# 

import pandas as pd

activity_file = 'encode_roadmap_act.txt'
# activity_file = 'encode_roadmap_acc_50k.txt'

activity_df = pd.read_table(activity_file, index_col=0)


# First, let's ask some sequence-centric questions. If we compute the proportion of active targets for each sequence, what does the distribution of this stat look like?

import matplotlib.pyplot as plt
import seaborn as sns
sns.set(font_scale=1.3)

seq_activity = activity_df.mean(axis=1)

constitutive_pct = sum(seq_activity > 0.5) / float(seq_activity.shape[0])
print '%.4f constitutively active sequences' % constitutive_pct

sns.distplot(seq_activity, kde=False)


cell_activity = df.mean(axis=0)

ca_out = open('cell_activity.txt', 'w')
for ci in range(len(cell_activity)):
    cols = (str(ci), df.columns[ci], str(cell_activity[ci]))
    print >> ca_out, '\t'.join(cols)
ca_out.close()

print cell_activity.min(), cell_activity.max()
print cell_activity.median()

sns.distplot(cell_activity, kde=False)


# construct matrix
X = np.array(df).T

print X.shape

# dimensionality reduction
model = Isomap(n_components=2, n_neighbors=10)
X_dr = model.fit_transform(X)


# plot PCA
plt.figure(figsize=(16,12), dpi=100)
plt.scatter(X_dr[:,0], X_dr[:,1], c='black', s=3)
#plt.ylim(-10,15)
#plt.xlim(-14,15)

for label, x, y in zip(df.columns, X_dr[:,0], X_dr[:,1]):
    plt.annotate(label, xy=(x,y), size=10)
    
plt.tight_layout()
plt.savefig('pca.pdf')


# Isomap dimensionality reduction
model = Isomap(n_components=2, n_neighbors=5)
X_dr = model.fit_transform(X)


# plot
plt.figure(figsize=(16,12), dpi=100)
plt.scatter(X_dr[:,0], X_dr[:,1], c='black', s=3)
#plt.ylim(-10,15)
#plt.xlim(-14,15)

for label, x, y in zip(df.columns, X_dr[:,0], X_dr[:,1]):
    plt.annotate(label, xy=(x,y), size=10)
    
plt.tight_layout()
plt.savefig('isomap.pdf')


t0 = time.time()

seq_samples = random.sample(xrange(X.shape[1]), 1000)

sns.set(font_scale=0.6)
plt.figure()
sns.clustermap(df.iloc[seq_samples].T, metric='jaccard', cmap='Reds', linewidths=0, xticklabels=False, figsize=(13,18))
plt.savefig('clustermap.pdf')

print 'Takes %ds' % (time.time() - t0)


# Saturated mutagenesis is a powerful tool both for dissecting a specific sequence of interest and understanding what the model learned. *basset_sat.py* enables this analysis from a test set of data.
# 
# To run this tutorial, you'll need to either download the pre-trained model from https://www.dropbox.com/s/rguytuztemctkf8/pretrained_model.th.gz and preprocess the consortium data, or just substitute your own files here:
# 

model_file = '../data/models/pretrained_model.th'
seqs_file = '../data/encode_roadmap.h5'


# This analysis is somewhat time consuming, proportional to the number of sequences, number of center nucleotides mutated, and number of plots. Typically, I just focus on one or a few targets of interest and provide a few hundred sequences.
# 
# Then we run the script as follows:
# * -t specifies which target indexes to plot
# * -n specifices the number of center nucleotides to mutate
# * -s samples 10 sequences
# * -o specifies the output directory
# 

import subprocess

cmd = 'basset_sat.py -t 46 -n 200 -s 10 -o satmut %s %s' % (model_file, seqs_file)
subprocess.call(cmd, shell=True)


# Then I often just browse through and check out the various sequences. In this set of 10, there isn't a whole lot going on. (It's a big genome.) But this one contains a weak CTCF motif, which is highlighted by loss scores. Note that the motif also contains a few positions marked by high gain scores where CTCF binding and accessibility would increase after mutation.
# 

# actual file is satmut/chr17_4904020-4904620\(+\)_c46_heat.pdf

from IPython.display import Image
Image(filename='satmut_eg/chr17_4904020-4904620(+)_c46_heat.png')


# If you have a particular sequence of interest, you can also provide *basset_sat.py* with a FASTA file in place of the HDF5 test set.
# 
# To demonstrate, I chose a [fascinating boundary domain, separating rostral from caudal gene expression programs in the HOXA locus](http://www.sciencemag.org/content/347/6225/1017.full). The exact region that I grabbed is chr7:27183235-27183835.
# 
# By specifying -1 to -t, we'll print heat maps for all cells.
# 

cmd = 'basset_sat.py -t -1 -n 200 -o satmut_hox %s satmut_eg/hoxa_boundary.fa' % model_file
subprocess.call(cmd, shell=True)


# As mentioned in the paper, the CTCF motif here does not match the consensus; a C->T mutation would cause it to match far better. Accordingly, Basset does not predict high accessibility in any cell type, but does highlight the motif and a GC-rich region to it's left.
# 

# actual file is satmut_hox/chr7_27183235-27183835_c127_heat.pdf

Image(filename='satmut_eg/chr7_27183235-27183835_c127_heat.png')


