# Do not let @jennybryan see this
setwd("/home/azureuser/cloudfiles/code/ignite-brk2019/R")

library(azuremlsdk)
library(ggplot2)

## Connect to saved Azure ML workspace
ws = load_workspace_from_config()


## Load "attrition" data from Datasets as an R data frame
## Data source: https://www.kaggle.com/pavansubhasht/ibm-hr-analytics-attrition-dataset
attrition = ws$datasets$`IBM-Employee-Attrition`
mc = attrition$to_csv_files()$mount()
mc$start()
csv_file = paste(mc$mount_point, 'part-00000', sep = '/')
df = read.csv(csv_file)
mc$stop()

## Clean up imported data
df$Attrition = as.factor(df$Attrition)
df$Education = as.factor(df$Education)
df$Over18=as.factor(df$Over18)
df$OverTime=as.factor(df$OverTime)
df$EnvironmentSatisfaction = ordered(df$EnvironmentSatisfaction)
df$JobInvolvement=ordered(df$JobInvolvement)
df$JobSatisfaction=ordered(df$JobSatisfaction)
df$PerformanceRating=ordered(df$PerformanceRating)
df$RelationshipSatisfaction=ordered(df$RelationshipSatisfaction)
df$WorkLifeBalance=order(df$WorkLifeBalance)


# View summary of data
View(df)
summary(df)

gg = ggplot(df, aes(x=Age, y=MonthlyIncome)) +
  geom_point(size=0.5, color='steelblue') +
  geom_smooth(aes(),method="loess") +
  facet_grid(Department ~ Attrition) 
print(gg)
ggsave(gg, file="attrition.png")
