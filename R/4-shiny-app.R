# Load the ggplot2 package which provides
# the 'mpg' dataset.
library(ggplot2)
library(shiny)
library(azuremlsdk)
library(jsonlite)
library(data.table)

ws <- load_workspace_from_config()
all_data <- fread('../data/IBM-Employee-Attrition.csv',stringsAsFactors = TRUE)
# remove useless fields 
all_data = within(all_data, rm(EmployeeCount, Over18, StandardHours, EmployeeNumber))
# make sure attrition is a factor
for (col in c('Attrition')) 
  set(all_data, j=col, value=as.factor(all_data[[col]]))

# get service from the workspace to refresh the object
#service = ws$webservices$attritionr
#service = $attritionpython
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
                                      selection = 'single')
  
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




