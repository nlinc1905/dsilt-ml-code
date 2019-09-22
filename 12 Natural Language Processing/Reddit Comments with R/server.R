setwd('dsilt-ml-code/12 Natural Language Processing/Reddit Comments with R')

#install.packages(c('DBI', 'RSQlite', 'tm', 'wordcloud', 'RColorBrewer', 'ggplot2', 'shiny', 'shinydashboard'))
library(DBI)
library(RSQLite)
library(tm)
library(wordcloud)
library(RColorBrewer)
library(ggplot2)
library(shiny)
library(shinydashboard)

#--------------------------------------------------------------------------#
#------------------------Standard R Code-----------------------------------#
#--------------------------------------------------------------------------#

set.seed(12)

#Specify sqlite driver
sqlDriv <- dbDriver("SQLite")
#Connect to sqlite db file
conn <- dbConnect(drv=sqlDriv, dbname="redditcomments_may2015.sqlite")

#Create color palette for word cloud
wcpal <- brewer.pal(9, "YlGnBu")
wcpal <- wcpal[-(1:4)]

#Create a function to find associated words for a given word and desired minimum correlation
fass <- function(inputData, word, min_correlation, max_related_words) {
  correlations <- findAssocs(inputData, word, min_correlation)[[1]]
  fass_df <- data.frame(corr=correlations,
                        terms=names(correlations))
  fass_df$terms <- factor(fass_df$terms, levels=fass_df$terms)
  #Sort data frame and scrap all but the top x
  fass_df <- fass_df[with(fass_df, order(corr, decreasing=T)),]
  if (nrow(fass_df) > max_related_words) {
    fass_df <- fass_df[1:max_related_words,]
  }
  return(fass_df)
}

#Create a function to build a data frame with a single word's frequency over time
fot <- function(inputData, word) {
  #Store the row from the terms matrix corresponding to a desired word, then convert to data frame
  item1 <- inspect(inputData[,which(colnames(inputData) %in% word)])
  df1 <- data.frame(comm_date=rownames(item1), word_in_comm=item1[,1])
  
  #Total the word frequencies by date
  totals_by_date <- aggregate(df1[,2], by=list(df1[,1]), FUN=sum)
  colnames(totals_by_date) <- c('COMM_DATE', 'NUMBER_OF_COMMENTS')
  totals_by_date$COMM_DATE <- as.Date(totals_by_date$COMM_DATE, '%Y-%m-%d')
  return(totals_by_date)
}

runQuery <- function(startdt, enddt) {
  message("Running query...")
  #Read the queries from files to reduce clutter and make it easier to edit the queries
  q_comments <- readChar("get_comments.sql", file.info("get_comments.sql")$size)
  
  #Insert date range for comment search
  q_comments <- gsub("&Date1", paste('\'', startdt, '\'', sep=''), q_comments)
  q_comments <- gsub("&Date2", paste('\'', enddt, '\'', sep=''), q_comments)
  
  #Run the query
  comms <- dbGetQuery(conn, q_comments)
  #Regex to remove non-alphanumeric characters
  comms$body <- gsub("[^[:alnum:] ]", "", comms$body)
  comms$created_date <- as.Date(comms$created_utc, '%Y-%m-%d')
  #Sort by date/time and then by author
  comms <- comms[with(comms, order(comms$created_utc, comms$author)),]
  
  message("Generating corpus and term matrix...")
  commscorpus <- Corpus(VectorSource(comms$body))
  commscorpus <- tm_map(commscorpus, content_transformer(tolower))
  commscorpus <- tm_map(commscorpus, removePunctuation)
  commscorpus <- tm_map(commscorpus, PlainTextDocument)
  commscorpus <- tm_map(commscorpus, removeWords, stopwords('english'))
  
  #These words appeared frequently and were determined to be useless for word cloud
  useless_terms <- c('and', 'but', 'has', 'had', 'how', 'was', 'got',
                     'did', 'didnt', 'are', 'arent', 'isnt', 'is',
                     'these', 'those', 'that', 'this', 'which', 
                     'their', 'theirs', 'they', 'we', 'us', 'ours', 
                     'my', 'mine', 'ive', 'id', 'im', 'than', 'theres',
                     'wouldve', 'couldve', 'wont', 'will', 'would',
                     'could', 'what', 'dont', 'you', 'can', 'with', 
                     'without', 'who', 'its', 'have', 'for', 'from',
                     'just', 'thats', 'wouldnt', 'our', 'the', 'a',
                     'there', 'here', 'then', 'where', 'why', 'though',
                     'should', 'also', 'as', 'in', 'your', 'you',
                     'when', 'doesnt', 'does', 'were', 'youre', 'hes',
                     'wasnt', 'shes')
  commscorpus <- tm_map(commscorpus, removeWords, useless_terms)
  
  #Create a terms matrix
  commstm <- DocumentTermMatrix(commscorpus)
  
  #Create a terms matrix with comment entry dates as the row names (tm created from corpus where dates are documents)
  docReader <- readTabular(mapping=list(content='body', id='created_date'))
  commscorpus_w_dates <- VCorpus(DataframeSource(comms), readerControl=list(reader=docReader))
  commstm_w_dates <- DocumentTermMatrix(commscorpus_w_dates)
  
  #Generate results: a list of 4 data elements and inform user that charts are now being drawn
  message('Drawing charts...')
  results <- list(comms, commscorpus, commstm, commstm_w_dates)
  return(results)
}

#--------------------------------------------------------------------------#
#------------------------Shiny Server Code---------------------------------#
#--------------------------------------------------------------------------#

#Define server logic
shinyServer(function(input, output) {

  #Set a reactive function to run the query
  queryFunction <- reactive({
    runQuery(input$dates[1], input$dates[2])
  })
  
  #When clicked, renders output table by running query function, which runs query
  observe({
    if (input$execute >0) {
      x <- queryFunction()
      #Render data table with search ability, pagination, and results per page options
      output$table1 <- renderDataTable({
        x[[1]]
      })
      #Render word cloud
      output$txtCloud <- renderPlot({
        wordcloud(x[[2]],
                  scale=c(4,1),
                  colors=wcpal,
                  min.freq=input$slider1,
                  max.words=input$slider2)
      })
      #Render associated words plot
      output$assWords <- renderPlot({
        a_word <- fass(x[[3]], req(input$sWord1), input$slider3, input$nbrTerms)
        ggplot(a_word, aes(x=terms)) + geom_bar(aes(y=corr), data=a_word, stat="identity", width=0.5, fill="#56B4E9") + coord_flip() + scale_y_continuous(expand=c(0,0)) + labs(y="Correlation", x="Term") + ggtitle(paste("Terms Appearing with the Word ", "\"", input$sWord1, "\"", sep="")) + theme(text=element_text(size=20), plot.title=element_text(color="#666666", face="bold", size=20), axis.title=element_text(color="#666666", face="bold", size=20))
      })
      #Render word frequency over time plot
      output$wordFreq <- renderPlot({
        f_word <- fot(x[[4]], req(input$sWord2))
        xscale <- pretty(as.Date(f_word$COMM_DATE, '%Y-%m-%d'), 3)
        ggplot(f_word, aes(x=COMM_DATE)) + geom_point(aes(y=NUMBER_OF_COMMENTS), data=f_word, size=3, color='red', alpha=1/4) + scale_x_date(breaks=xscale, labels=xscale) + labs(y='Number of Comments', x='Date') + ggtitle(paste("Comments Containing the Word ", "\"", input$sWord2, "\"", " Over Time", sep="")) + theme(text=element_text(size=20), plot.title=element_text(color="#666666", face="bold", size=20), axis.title=element_text(color="#666666", face="bold", size=16))
      })
      #Link to download the selected dataset
      output$downloadData <- downloadHandler(
        filename=function() { paste("Reddit_Comments_May2015.csv", sep="") },
        content=function(filename) {
          write.csv(x[[1]], filename, row.names=F)
        })
    }
  })
})