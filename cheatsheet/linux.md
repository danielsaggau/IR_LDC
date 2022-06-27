# tmux commands 

## Start a new session 
`tmux new` 

## start a new session w session-name 
tmux new -s sessionname 

## list sessions 

tmux ls 



# slurm commands 

The main difference is that srun is interactive and blocking 
(you get the result in your terminal and 
you cannot write other commands until it is finished),
while sbatch is batch processing and non-blocking 
(results are written to a file and you can submit other commands right away).
The main difference is that srun is interactive and blocking 
(you get the result in your terminal and you cannot write other commands until it is finished),
while sbatch is batch processing and non-blocking (results are written to a file and you can submit
other commands right away).
