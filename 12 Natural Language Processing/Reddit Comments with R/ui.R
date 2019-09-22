setwd('dsilt-ml-code/12 Natural Language Processing/Reddit Comments with R')

library(shiny)
library(shinydashboard)

#Define UI for application
dashboardPage(
  dashboardHeader(title="May 2015 'NFL Draft' Subreddit Comment Analysis", titleWidth=500),
  dashboardSidebar(
    dateRangeInput("dates", label=h4("Comment Date Range"),
                   start='2015-05-01', end='2015-05-31',
                   min='2015-05-01', max='2015-05-31'),
    br(),
    actionButton("execute", label="Run", width=75)
  ),
  dashboardBody(
    fluidRow(
      tabBox(tabPanel(title="Word Cloud",
                      fluidRow(
                        box(title="Word Cloud", status="primary", solidHeader=T,
                            div(style='overflow-x:scroll;overflow-y:scroll',
                                plotOutput('txtCloud', height=400, width=600))),
                        box(title="Controls", status="primary", solidHeader=T,
                            sliderInput("slider1", label="Minimum Word Frequency",
                                        min=50, max=150, value=100),
                            sliderInput("slider2", label="Maximum Number of Words",
                                        min=500, max=1500, value=1000)))),
             tabPanel(title="Associated Words",
                      fluidRow(
                        box(title="Associated Cloud", status="primary", solidHeader=T,
                            div(style='overflow-x:scroll;overflow-y:scroll',
                                plotOutput('assWords', height=400, width=600))),
                        box(title="Controls", status="primary", solidHeader=T,
                            textInput("sWord1", label=h4("Word to Search"), value="cheating"),
                            sliderInput("slider3", label="Minimum Correlation",
                                        min=0, max=1, value=0.3),
                            numericInput("nbrTerms", label=h4("Maximum Associated Terms"), 
                                                              value=20)))),
             tabPanel(title="Word Frequency Over Time",
                      fluidRow(
                        box(title="Word Frequency Over Time", status="primary", solidHeader=T,
                            div(style='overflow-x:scroll;overflow-y:scroll',
                                plotOutput('wordFreq', height=400, width=600))),
                        box(title="Controls", status="primary", solidHeader=T,
                            textInput("sWord2", label=h4("Word to Search"), value="best")))),
             width=12)
    ),
    fluidRow(
      box(title="DataOutput", status="primary", solidHeader=T,
          downloadButton('downloadData', 'Download CSV'),
          br(),
          div(style='overflow-x:scroll', dataTableOutput("table1")), width=12))
  )
)