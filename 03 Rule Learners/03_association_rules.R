# https://archive.ics.uci.edu/ml/datasets/online+retail

setwd("dsilt-ml-code/03 Rule Learners")

#install.packages("openxlsx", dependencies = TRUE)
library(openxlsx)
library(Matrix)
library(arules)
library(arulesViz)

set.seed(14)

#-------------------------------------------------------------------------------------------------#
#---------------------------------------EDA and Cleaning------------------------------------------#
#-------------------------------------------------------------------------------------------------#

raw_d <- read.xlsx('Online Retail.xlsx', sheet='Online Retail')
head(raw_d)

naCol <- function(x){
  y <- sapply(x, function(y) sum(length(which(is.na(y) | is.nan(y) | is.infinite(y) | y=='NA' | y=='NaN' | y=='Inf' | y==''))))
  y <- data.frame('feature'=names(y), 'count.nas'=y)
  row.names(y) <- c()
  y
}

naCol(raw_d)
# Documentation says InvoiceNo starting with "C" is cancellation
nrow(raw_d[substring(as.character(raw_d$InvoiceNo), 1, 1)=='C',])
unique(substring(as.character(raw_d$InvoiceNo), 1, 1))  # What does "A" mean?
raw_d[substring(as.character(raw_d$InvoiceNo), 1, 1)=="A", 1:6]

# Are there any transactions with unusual quantities?
table(raw_d$Quantity)
raw_d[raw_d$Quantity<0, 1:6]    # Possibly returns
raw_d[raw_d$Quantity>1000, 1:6] # Cheap items in bulk

# Are there any unusual unit prices?
hist(raw_d$UnitPrice)
raw_d[raw_d$UnitPrice<0, 1:6]    # Adjustments
raw_d[raw_d$UnitPrice>1000, 1:6] # Postage and Amazon Fees
# Consider removing StockCodes in ['AmazonFee', 'M' , 'Post', 'Dot', B']

# Can a customer have multiple invoices?
aggregate(InvoiceNo ~ CustomerID, data=raw_d, FUN=length)

# Does an invoice contain multiple items?
items_by_invoice <- aggregate(StockCode ~ InvoiceNo, data=raw_d, FUN=length)
items_by_invoice
# Are there invoices with only 1 item?  These can be removed.
invoices_to_remove <- items_by_invoice[items_by_invoice$StockCode<2, 'InvoiceNo']

# Remove data that could produce meaningless or useless rules
clean_d <- raw_d[!(substring(as.character(raw_d$InvoiceNo), 1, 1) %in% c('C', 'A'))
                 & (raw_d$Quantity>0)
                 & (raw_d$UnitPrice>0)
                 & !(raw_d$StockCode %in% c('AMAZONFEE', 'M', 'POST', 'DOT', 'B'))
                 & !(raw_d$InvoiceNo %in% invoices_to_remove),
                 c('InvoiceNo', 'StockCode', 'Description')]
clean_d <- clean_d[complete.cases(clean_d),]

sapply(clean_d, class)
clean_d$InvoiceNo <- as.integer(clean_d$InvoiceNo)
clean_d$Description <- trimws(clean_d$Description)

write.csv(clean_d, 'online_retail_clean.csv', row.names=F)

#-------------------------------------------------------------------------------------------------#
#------------------------------------Apriori Association Rules------------------------------------#
#-------------------------------------------------------------------------------------------------#

# Cast to sparse matrix
trxns <- as(split(clean_d[, "Description"], 
                  clean_d[, "InvoiceNo"]), 
            "transactions")
dm <- as(trxns, "ngCMatrix")  # ngCMatrix is more efficient than dgCMatrix

# Sparse matrix sense checks
length(unique(clean_d$Description)) #should match nbr rows
length(unique(clean_d$InvoiceNo))   #should match nbr columns
str(trxns)
str(dm)
sum(colSums(dm, na.rm=F, dims = 1, sparseResult=F)<1)
sum(colSums(dm, na.rm=F, dims = 1, sparseResult=F)==Inf)
sum(rowSums(dm, na.rm=F, dims = 1, sparseResult=F)<1)
sum(rowSums(dm, na.rm=F, dims = 1, sparseResult=F)==Inf)

# Inspect part of the matrix and look for any obvious patterns
arules::image(dm[1:2000, 1:2000])

#Default hyperparameters: minimum sup=0.5 and conf=0.8 were not satisfactory
dmrules <- apriori(dm, parameter=list(sup=0.03, conf=0.1, target="rules", minlen=2))
inspect(dmrules)

dmrules <- apriori(dm, parameter=list(sup=0.02, conf=0.7, target="rules", minlen=2))
inspect(dmrules)

dmrules <- apriori(dm, parameter=list(sup=0.01, conf=0.8, target="rules", minlen=2))
inspect(head(sort(dmrules, by="lift"), 25))

dmrules_df <- as(sort(dmrules, by="lift"), "data.frame")
write.csv(dmrules_df, 'online_retail_learned_rules.csv', row.names=F)

# Visualizing rules
plot(dmrules, measure=c('support', 'lift'), shading='confidence', interactive=T)
plot(head(dmrules, 10), method='grouped')
plot(dmrules, method='graph')
plot(dmrules, method='graph', control=list(type='itemsets'))
plot(dmrules, method='paracoord')
#plot(dmrules, method='paracoord', control=list(reorder=T))  #Takes a long time to reorder
# Documentation here: https://rdrr.io/rforge/arulesViz/f/inst/doc/arulesViz.pdf
