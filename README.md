# GA-project

Simple genetic algorithm used to study thermodynamic properties of
evolutionary processes.  See `GA-entropy.tex` for a write-up of the
science and the conclusions.

## Running an evolution

### Pick parameters and start the run

You can run an evolution by adjusting the values of `mate_func`,
`initial_func`, and `fit_func` in the `main()` function of
`GA-poly-max.py`

Then you can run it with:

`./GA-poly-max.py`

The output will tell you what your output file name will be, for
example: `GA-gen-info_pid3554872.out`

### Plotting the evolution

You can (at any time during the run) get a view of what's happening
with a command like:

`./plot_ga_output.py GA-gen-info_pid3554872.out &`

You could also view *all* the output files in the current directory
with

`./plot_ga_output.py &`

or you can give a few `GA-gen-info_pid????.out` file names on the
command line and it will make plots for those files.  The plots are
shown interactively, and they are also saved as `.pdf` files.

## Building the paper

Right now we are using this figure as an example:

`GA-gen-info_pid3095791.out.pdf`

with that pdf file present you can run:

`latexmk -pdf GA-entropy.tex`

to generate GA-entropy.pdf

## Building the slides

latexmk -pdf GA-entropy-slides.tex

will generate GA-entropy-slides.pdf

## Prerequisites

The python packages we used for the code are:

- matplotlib for the graphs, 
- pandas for the data frame, 
- shortuuid for unique identifiying strings, 
- numpy for population creation and advanced math, 
- struct for binary editing, 
- os and sys for nan removal as well as file creation, 
- and math for regular arithmetic
