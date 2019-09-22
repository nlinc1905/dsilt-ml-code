library(networkDynamicData)
library(sqldf)

data("enronEmails")
# Filter to valid dates (raw data has some incorrect timestamps)
enronEmails <- network.extract(enronEmails, 
                               onset=883612800, 
                               terminus=1024688419)

d <- as.data.frame(enronEmails)
d <- d[d$tail != d$head,] # removes emails to oneself
d <- d[,c('onset', 'tail', 'head')]
colnames(d) <- c('timestamp', 'to_id', 'from_id')
d$timestamp <- as.POSIXct(d$timestamp, origin="1970-01-01")

node_names <- get.vertex.attribute(enronEmails, 'email_id')
node_names <- data.frame(name=node_names)
node_names$node_id <- seq.int(nrow(node_names))

# Join node names to nodes
query <- "
SELECT d.*, nfrom.name AS from_name, nto.name AS to_name
FROM d
LEFT JOIN node_names nfrom ON nfrom.node_id = d.from_id
LEFT JOIN node_names nto ON nto.node_id = d.to_id
"
d <- sqldf(query)

# Re-number the node IDs
temp <- data.frame(original_node_id=unique(c(d$from_id, d$to_id)))
temp$node_id <- seq.int(nrow(temp))
query <- "
SELECT d.timestamp, tempfrom.node_id AS from_id, tempto.node_id AS to_id, d.from_name, d.to_name
FROM d
LEFT JOIN temp tempfrom ON tempfrom.original_node_id = d.from_id
LEFT JOIN temp tempto ON tempto.original_node_id = d.to_id
"
d <- sqldf(query)

write.csv(d, file="dsilt-ml-code/16 Network Analysis/enron_emails_network.csv", row.names=F)
