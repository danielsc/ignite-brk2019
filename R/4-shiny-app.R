## Shiny app to test predictions of all deployed
## attrition models
## Try running this in your local environment!
library(ggplot2)
library(shiny)
library(azuremlsdk)
library(jsonlite)
library(data.table)

ws <- load_workspace_from_config()

## Get data to use as prediction rows
all_data <- fread('../data/IBM-Employee-Attrition.csv',stringsAsFactors = TRUE)
# remove useless fields 
all_data = within(all_data, rm(EmployeeCount, Over18, StandardHours, EmployeeNumber))
# make sure attrition is a factor
all_data$Attrition = as.factor(all_data$Attrition)

# get service from the workspace to refresh the object
webservices = ws$webservices
webservices_string = lapply(webservices, function(s) s$name)

# Define UI for slider demo app ----
ui <- fluidPage(
  titlePanel("Score AzureML Model"),
  selectInput("model", "Pick a model:", choices = webservices_string),
  # Create a new row for the table.
  div(DT::dataTableOutput("table"), style = "font-size: 75%; width: 75%"),
  h4("JSON sent to service"),
  verbatimTextOutput("json"),
  h4("Result returned by service"),
  verbatimTextOutput('result')
)

# Define server logic for slider examples ----
server <- function(input, output) {
  
  # Filter data based on selections
  output$table <- DT::renderDataTable(all_data,
                                      selection = 'single',
                                      options=list(pageLength=5))
  
  output$json <- renderText({
    if (is.null(input$table_rows_selected)) {
      "select a table row"
    } else {
      sample = all_data[input$table_rows_selected]
      sample$Attrition = NULL
      toJSON(sample)
    }
  })
  
  output$result <- renderText({
    if (is.null(input$table_rows_selected)) {
      "select a table row"
    } else {
      sample = all_data[input$table_rows_selected]
      sample$Attrition = NULL
      service = webservices[[input$model]]
      result = invoke_webservice(service, toJSON(sample))
      toString(result)
    }
  })
  
  
}

# Create Shiny app ----
shinyApp(ui, server)
