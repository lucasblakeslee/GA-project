# GA-project

Simple genetic algorithm used to study thermodynamic properties of
evolutionary processes.  See GA-entropy.tex for a write-up of the
science and the conclusions.

## Building the paper

Right now we are using this figure as an example:

GA-gen-info_pid3095791.out.pdf

with that pdf file present you can run:

latexmk -pdf GA-entropy.tex

to generate GA-entropy.pdf

## Prerequisites

The python packages we used for the code are:
* matplotlib for the graphs, 
* pandas for the data frame, 
* shortuuid for unique identifiying strings, 
* numpy for population creation and advanced math, 
* struct for binary editing, 
* os and sys for nan removal as well as file creation, 
* and math for regular arithmetic
