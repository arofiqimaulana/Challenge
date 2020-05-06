library(flexdashboard)
library(shiny)

wd <- file.choose()

library(xlsx)
df <- read.xlsx(wd,header = TRUE,sheetName = "Sheet1")
head(df)

install.packages("rcdimple")
library(rcdimple)
install.packages("highcharter")

library(sf)
library(ggplot2)
library(tmap)
library(tmaptools)
library(leafleat)
library(dplyr)
library(highcharter)

tm_shape(wd)
