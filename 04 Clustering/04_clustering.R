
setwd("dsilt-ml-code/04 Clustering")

library(readxl)
library(ggplot2)
library(dplyr)
library(cluster)
library(dbscan)
library(Rtsne)

set.seed(14)

#-------------------------------------------------------------------------------------------------#
#---------------------------------------EDA and Cleaning------------------------------------------#
#-------------------------------------------------------------------------------------------------#

raw_d <- read_excel('Online Retail.xlsx', sheet='Online Retail')
head(raw_d)

# Add basic features
raw_d$Country <- as.factor(raw_d$Country)
raw_d$InvoiceDate <- as.POSIXct(raw_d$InvoiceDate)
raw_d$Date <- format(raw_d$InvoiceDate,"%Y-%m-%d")
raw_d$Time <- format(raw_d$InvoiceDate,"%H:%M:%S")
raw_d$Hour <- as.integer(format(raw_d$InvoiceDate,"%H"))
paste(min(raw_d$Date), max(raw_d$Date)) #Should match documentation
raw_d$Amount <- as.numeric(raw_d$Quantity*raw_d$UnitPrice)
raw_d <- transform(raw_d, InvoiceNbrItems=as.numeric(ave(StockCode, InvoiceNo, FUN=length)))
raw_d <- transform(raw_d, InvoiceQuantity=ave(Quantity, InvoiceNo, FUN=sum))
raw_d <- transform(raw_d, InvoiceTotal=ave(UnitPrice*Quantity, InvoiceNo, FUN=sum))
raw_d$Day <- as.integer(format(raw_d$InvoiceDate,"%d"))
raw_d$Month <- as.integer(format(raw_d$InvoiceDate,"%m"))
raw_d$Year <- as.integer(format(raw_d$InvoiceDate,"%Y"))

naCol <- function(x){
  y <- sapply(x, function(y) sum(length(which(is.na(y) 
                                              | is.nan(y) 
                                              | is.infinite(y) 
                                              | y=='NA' 
                                              | y=='NaN' 
                                              | y=='Inf' 
                                              | y==''))))
  y <- data.frame('feature'=names(y), 'count.nas'=y)
  row.names(y) <- c()
  y
}

#naCol can't handle POSIXct, but doesn't need to here
naCol(raw_d[,-which(colnames(raw_d) %in% c('InvoiceDate'))])
sapply(raw_d, class)

# Remove some of the same records as before (see chapter 3)
clean_d <- raw_d[!(substring(as.character(raw_d$InvoiceNo), 1, 1) %in% c('C', 'A'))
                 & (raw_d$Quantity>0)
                 & (raw_d$UnitPrice>0)
                 & !(raw_d$StockCode %in% c('AMAZONFEE', 'M', 'POST', 'DOT', 'B'))
                 & !(is.na(raw_d$Description)),]
# Before removing missing customer IDs, see if they share a pattern
clean_d$MissingCust <- as.factor(ifelse(is.na(clean_d$CustomerID), 1, 0))
ggplot(clean_d[clean_d$Quantity<50,], aes(x=MissingCust, y=Quantity, fill=MissingCust)) +
  geom_boxplot()
ggplot(clean_d[clean_d$InvoiceTotal<1000,], aes(x=MissingCust, y=InvoiceTotal, fill=MissingCust)) +
  geom_boxplot()
# Missing custs buy fewer items of each type but have larger average bills

clean_d$InvoiceNo <- as.integer(clean_d$InvoiceNo)
clean_d$Description <- trimws(clean_d$Description)

# Explore countries
sort(table(clean_d$Country), decreasing=T)
barplot(sort(table(clean_d$Country), decreasing=T), 
        main="Number of Transactions by Country")
ggplot(clean_d, aes(x=Country, y=Quantity, fill=Country)) +
  geom_boxplot()
ggplot(clean_d[clean_d$Quantity<150,], aes(x=Country, y=Quantity, fill=Country)) +
  geom_boxplot() + coord_flip()

# Look for outliers
table(clean_d$UnitPrice)
hist(clean_d$Amount)
clean_d[clean_d$Amount>10000,]
hist(clean_d[clean_d$Amount<1000, 'InvoiceTotal'])
clean_d[clean_d$InvoiceTotal>50000,]
table(clean_d$Quantity)
clean_d[clean_d$Quantity>5000,]
# No outliers appear to be data errors

#-------------------------------------------------------------------------------------------------#
#--------------------------------------Feature Engineering----------------------------------------#
#-------------------------------------------------------------------------------------------------#

# Separate unknown customers in case it is useful later on
unknown_custs <- clean_d[clean_d$MissingCust==1,]
unknown_custs$MissingCust <- NULL
clean_d <- clean_d[clean_d$MissingCust==0,]
clean_d$MissingCust <- NULL

sapply(clean_d, class)

# Create customer level features
clean_d <- transform(clean_d, CustNumInvoices=ave(InvoiceNo, CustomerID, FUN=length))

clean_d <- transform(clean_d, CustTotalItems=as.integer(ave(StockCode, CustomerID, FUN=length)))
clean_d <- transform(clean_d, CustAvgItems=as.numeric(ave(InvoiceNbrItems, CustomerID, FUN=mean)))
clean_d <- transform(clean_d, CustStdItems=as.numeric(ave(InvoiceNbrItems, CustomerID, FUN=sd)))
clean_d <- transform(clean_d, CustMinItems=as.numeric(ave(InvoiceNbrItems, CustomerID, FUN=min)))
clean_d <- transform(clean_d, CustMaxItems=as.numeric(ave(InvoiceNbrItems, CustomerID, FUN=max)))
clean_d$CustItemsRange <- clean_d$CustMaxItems-clean_d$CustMinItems

clean_d <- transform(clean_d, CustTotalQuant=as.numeric(ave(Quantity, CustomerID, FUN=sum)))
clean_d <- transform(clean_d, CustAvgQuant=as.numeric(ave(InvoiceQuantity, CustomerID, FUN=mean)))
clean_d <- transform(clean_d, CustStdQuant=as.numeric(ave(InvoiceQuantity, CustomerID, FUN=sd)))
clean_d <- transform(clean_d, CustMinQuant=as.numeric(ave(InvoiceQuantity, CustomerID, FUN=min)))
clean_d <- transform(clean_d, CustMaxQuant=as.numeric(ave(InvoiceQuantity, CustomerID, FUN=max)))
clean_d$CustQuantRange <- clean_d$CustMaxQuant-clean_d$CustMinQuant

clean_d <- transform(clean_d, CustTotalAmt=as.numeric(ave(UnitPrice, CustomerID, FUN=sum)))
clean_d <- transform(clean_d, CustAvgAmt=as.numeric(ave(InvoiceTotal, CustomerID, FUN=mean)))
clean_d <- transform(clean_d, CustStdAmt=as.numeric(ave(InvoiceTotal, CustomerID, FUN=sd)))
clean_d <- transform(clean_d, CustMinAmt=as.numeric(ave(InvoiceTotal, CustomerID, FUN=min)))
clean_d <- transform(clean_d, CustMaxAmt=as.numeric(ave(InvoiceTotal, CustomerID, FUN=max)))
clean_d$CustAmtRange <- clean_d$CustMaxAmt-clean_d$CustMinAmt

clean_d <- transform(clean_d, CustAvgHr=as.numeric(ave(Hour, CustomerID, FUN=mean)))
clean_d <- transform(clean_d, CustStdHr=as.numeric(ave(Hour, CustomerID, FUN=sd)))
clean_d <- transform(clean_d, CustMinHr=as.numeric(ave(Hour, CustomerID, FUN=min)))
clean_d <- transform(clean_d, CustMaxHr=as.numeric(ave(Hour, CustomerID, FUN=max)))
clean_d$CustHrRange <- clean_d$CustMaxHr-clean_d$CustMinHr

temp <- clean_d %>%
  select(CustomerID, InvoiceNo, Date) %>%
  arrange(CustomerID, Date) %>%
  group_by(CustomerID) %>%
  mutate(RunDaysSinceLastPurch=ifelse(!is.na(lag(Date, 1)), difftime(Date, lag(Date, 1), units="days"), NA))
clean_d <- cbind(clean_d, as.data.frame(temp)[,'RunDaysSinceLastPurch'])
rm(temp)
colnames(clean_d)[ncol(clean_d)] <- 'RunDaysSinceLastPurch'

clean_d <- transform(clean_d, AvgDaysBtwnPurch=as.numeric(ave(RunDaysSinceLastPurch, CustomerID, FUN=(function(x) {mean(x, na.rm=T)}))))
clean_d <- transform(clean_d, StdDaysBtwnPurch=as.numeric(ave(RunDaysSinceLastPurch, CustomerID, FUN=(function(x) {sd(x, na.rm=T)}))))
clean_d <- transform(clean_d, MinDaysBtwnPurch=as.numeric(ave(RunDaysSinceLastPurch, CustomerID, FUN=(function(x) {min(x, na.rm=T)}))))
clean_d <- transform(clean_d, MaxDaysBtwnPurch=as.numeric(ave(RunDaysSinceLastPurch, CustomerID, FUN=(function(x) {max(x, na.rm=T)}))))
clean_d$DaysBtwnPurchRange <- clean_d$MaxDaysBtwnPurch-clean_d$MinDaysBtwnPurch

# Market to new customers
clean_d <- transform(clean_d, CustFirstPurchaseDt=as.Date(ave(Date, CustomerID, FUN=min)))
clean_d$NewCust <- ifelse(difftime(as.Date(max(clean_d$Date)), clean_d$CustFirstPurchaseDt, units="days")<90, 1, 0)
# Entice lost customers to return
clean_d <- transform(clean_d, CustLastPurchaseDt=as.Date(ave(Date, CustomerID, FUN=max)))
clean_d$LostCust <- ifelse(difftime(clean_d$CustLastPurchaseDt, as.Date(min(clean_d$Date)), units="days")<90, 1, 0)
#Market to recent customers (not necessarily new)
clean_d$DaysSinceLastPurch <- as.integer(difftime(Sys.Date(), clean_d$CustLastPurchaseDt, units="days"))

clean_d <- clean_d[complete.cases(clean_d),]  #Small price to pay
head(clean_d)
write.csv(clean_d, 'online_retail_clean.csv', row.names=F)

#-------------------------------------------------------------------------------------------------#
#------------------------------------------Clustering---------------------------------------------#
#-------------------------------------------------------------------------------------------------#

cols_not_to_cluster <- c('InvoiceNo', 'StockCode', 'Description', 'Quantity', 
                         'InvoiceDate', 'UnitPrice', 'Country', 
                         'Date', 'Time', 'Hour', 'Amount', 'InvoiceNbrItems', 
                         'InvoiceQuantity', 'InvoiceTotal', 'Day', 'Month',
                         'Year', 'RunDaysSinceLastPurch', 'NewCust', 'LostCust')
d_to_clust <- clean_d[,-which(colnames(clean_d) %in% cols_not_to_cluster)]
d_to_clust$CustFirstPurchaseDt <- as.integer(d_to_clust$CustFirstPurchaseDt)
d_to_clust$CustLastPurchaseDt <- as.integer(d_to_clust$CustLastPurchaseDt)
d_to_clust <- unique(d_to_clust)

logTransform <- function(x) {
  return(log(x+0.001))  # Add small number to prevent Inf
}

standardize <- function(x) {
  return((x - mean(x))/sd(x))
}

deStandardize <- function(x) {
  return((x)*sd(x)+mean(x))
}

orig_col_order <- colnames(d_to_clust)
cols_to_log_transform <- c('CustNumInvoices', 'CustTotalItems', 'CustAvgItems', 
                           'CustStdItems', 'CustTotalQuant', 'CustAvgQuant', 
                           'CustStdQuant', 'CustTotalAmt', 'CustAvgAmt', 
                           'CustStdAmt', 'AvgDaysBtwnPurch', 'StdDaysBtwnPurch')
d_to_clust_std <- cbind(d_to_clust[,-which(colnames(d_to_clust) %in% cols_to_log_transform)],
                        as.data.frame(sapply(d_to_clust[,which(colnames(d_to_clust) 
                                                               %in% cols_to_log_transform)], 
                                             logTransform)))
d_to_clust_std <- d_to_clust_std[orig_col_order]
d_to_clust_std <- cbind(CustomerID=d_to_clust_std$CustomerID, 
                        as.data.frame(sapply(d_to_clust_std[,-1], standardize)))
d_to_clust_std[is.na(d_to_clust_std)] <- 0
head(d_to_clust_std)

# k-means
km <- kmeans(d_to_clust_std[,-1], centers=5, nstart=20)
km$tot.withinss
plot(d_to_clust_std[,c('CustNumInvoices', 'CustAvgAmt')], col=(km$cluster+1))

# partitioning around medoids
pm <- pam(d_to_clust_std[,-1], k=5, metric='euclidean')
pm
plot(d_to_clust_std[,c('CustNumInvoices', 'CustAvgAmt')], col=(pm$clustering))

# DBSCAN
#dsub <- d_to_clust_std[sample(nrow(d_to_clust_std), 10000),]
minpts <- ncol(d_to_clust_std[,-1])+1
kNNdistplot(as.matrix(d_to_clust_std[,-1]), k=minpts)
dbs <- dbscan(d_to_clust_std[,-1], eps=3, minPts=minpts)
dbs
plot(d_to_clust_std[,c('CustNumInvoices', 'CustAvgAmt')], col=(dbs$cluster+1))

# OPTICS
opt <- optics(d_to_clust_std[,-1], eps=3, minPts=minpts,  
              search="kdtree", bucketSize=10, splitRule="suggest")
plot(opt)
opt_cut <- extractXi(opt, xi=0.003)
plot(opt_cut)
hullplot(d_to_clust_std[,c('CustNumInvoices', 'CustAvgAmt')], opt_cut)
plot(d_to_clust_std[,c('CustNumInvoices', 'CustAvgAmt')], col=(opt_cut$cluster+1))

# t-SNE
tsne <- Rtsne(d_to_clust_std[,-1], dims=2, perplexity=30, max_iter=500)
plot(tsne$Y, col=(km$cluster+1), main='t-SNE with k-Means Clusters')
plot(tsne$Y, col=(pm$clustering), main='t-SNE with PAM Clusters')
plot(tsne$Y, col=(dbs$cluster+1), main='t-SNE with DBSCAN Clusters')
plot(tsne$Y, col=(opt_cut$cluster+1), main='t-SNE with OPTICS Clusters')

# recency, frequency, monetary segmentation here: 
# https://www.kaggle.com/mgmarques/customer-segmentation-and-market-basket-analysis

# Re-cluster data for more meaningful results
new_d_to_clust <- as.data.frame(sapply(d_to_clust[,which(colnames(d_to_clust) 
                                                         %in% c("CustAvgItems", 
                                                                "CustAvgAmt",
                                                                "AvgDaysBtwnPurch",
                                                                "DaysSinceLastPurch"))], 
                                       logTransform))
new_d_to_clust_std <- d_to_clust_std[,which(colnames(d_to_clust_std) 
                                            %in% c("CustAvgItems", 
                                                   "CustAvgAmt",
                                                   "AvgDaysBtwnPurch",
                                                   "DaysSinceLastPurch"))]
km <- kmeans(new_d_to_clust_std, centers=4, nstart=20)
plot(new_d_to_clust[,c("CustAvgAmt", "AvgDaysBtwnPurch")], col=(km$cluster+1))
plot(new_d_to_clust[,c("CustAvgItems", "CustAvgAmt")], col=(km$cluster+1))
plot(new_d_to_clust[,c("CustAvgAmt", "DaysSinceLastPurch")], col=(km$cluster+1))
tsne <- Rtsne(new_d_to_clust_std, dims=2, perplexity=30, max_iter=500)
plot(tsne$Y, col=(km$cluster+1), main='t-SNE with k-Means Clusters')
