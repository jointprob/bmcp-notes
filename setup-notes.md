+ Export environment - env named bmcp
  + `$mamba env export -n bmcp > environment.yml`
+ Create environment
  + `$mamba env create --file environment.yml`
+ Delete environment
  + `$mamba env remove --name bmcp`
+ List envs
  + `$mamba env list`
+ pip install `rpy2` so that it recognises the system R
